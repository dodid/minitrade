from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from minitrade.broker import Broker, BrokerAccount, OrderValidator
from minitrade.trader import TradePlan
from minitrade.utils.config import config
from minitrade.utils.mtdb import MTDB

if TYPE_CHECKING:
    from minitrade.backtest import RawOrder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class IbOrderLog:
    id: str
    order_id: str
    account_id: str
    plan: str
    order: str
    iborder: str
    result: str
    exception: str
    broker_order_id: str
    log_time: datetime


class InteractiveBrokers(Broker):

    def __init__(self, account: BrokerAccount):
        super().__init__(account)
        self._account_id = None
        self._admin_host = config.brokers.ib.gateway_admin_host
        self._admin_port = config.brokers.ib.gateway_admin_port
        self._port = None
        self._last_ready_check = None
        self._last_ready_status = False

    @property
    def account_id(self):
        if not self._account_id:
            accounts = self.get_account_info()
            if accounts:
                self._account_id = accounts[0]['id']
        return self._account_id

    def __call_ibgateway_admin(self, method: str, path: str, params: dict | None = None):
        '''Call the ibgateway's admin API'''
        url = f'http://{self._admin_host}:{self._admin_port}{path}'
        resp = requests.request(method=method, url=url, params=params)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code >= 400:
            raise RuntimeError(f'Request {path} returned {resp.status_code} {resp.text}')

    def __call_ibgateway(
            self, method: str, path: str, params: dict | None = None, json: Any | None = None, timeout: int = 20) -> Any:
        '''Call the ibgateway's REST API'''
        if self._port:
            url = f'http://localhost:{self._port}/v1/api{path}'
            resp = requests.request(method=method, url=url, params=params, json=json, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code >= 400:
                raise RuntimeError(f'Request {path} returned {resp.status_code} {resp.text}')
        else:
            raise RuntimeError(f'IB gateway port is not set for request {method} {path}')

    def is_ready(self) -> bool:
        if self._last_ready_check and datetime.now() - self._last_ready_check < timedelta(seconds=1):
            self._last_ready_check = datetime.now()
            return self._last_ready_status
        try:
            status = self.__call_ibgateway_admin('GET', f'/ibgateway/{self.account.alias}')
            if status and status['account'] == self.account.username:
                self._port = status['port']
                if status['authenticated']:
                    self._last_ready_check = datetime.now()
                    self._last_ready_status = True
                    return True
                else:
                    # if not authenticated, try reauthenticate
                    self.__call_ibgateway('POST', '/iserver/reauthenticate')
                    for _ in range(10):
                        status = self.__call_ibgateway_admin('GET', f'/ibgateway/{self.account.alias}')
                        if status and status['authenticated']:
                            self._port = status['port']
                            self._last_ready_check = datetime.now()
                            self._last_ready_status = True
                            return True
                        else:
                            time.sleep(1)
        except Exception:
            logger.exception(f'Checking broker status failed for {self.account.alias}')
        self._last_ready_check = datetime.now()
        self._last_ready_status = False
        return False

    def connect(self):
        if not self.is_ready():
            try:
                status = self.__call_ibgateway_admin('PUT', f'/ibgateway/{self.account.alias}')
                self._port = status['port']
            except Exception as e:
                raise ConnectionError(f'Login failed for {self.account.alias}') from e

    def disconnect(self):
        try:
            self.__call_ibgateway_admin('DELETE', f'/ibgateway/{self.account.alias}')
            self._port = None
        except Exception:
            logger.exception(f'Logout failed for IB account {self.account.alias}')

    def get_account_info(self) -> Any:
        accounts = self.__call_ibgateway('GET', '/portfolio/accounts')
        for account in accounts:
            account['ledger'] = self.__call_ibgateway('GET', f'/portfolio/{account["id"]}/ledger')
            account['performance'] = self.__call_ibgateway(
                'POST', f'/pa/performance', json={'acctIds': [account['id']], 'freq': 'D'})
        return accounts

    def get_portfolio(self) -> pd.DataFrame:
        portfolio = []
        for account in self.get_account_info():
            i = 0
            while True:
                page = self.__call_ibgateway('GET', f'/portfolio/{account["id"]}/positions/{i}')
                portfolio.extend(page)
                if len(page) < 100:
                    break
                else:
                    i += 1
        return pd.DataFrame(portfolio)

    def download_trades(self) -> pd.DataFrame:
        # the list may be incomplete, no definitive document is available
        whitelist = ['execution_id', 'symbol', 'supports_tax_opt', 'side', 'order_description', 'trade_time',
                     'trade_time_r', 'size', 'price', 'submitter', 'exchange', 'commission', 'net_amount', 'account',
                     'accountCode', 'company_name', 'contract_description_1', 'contract_description_2', 'sec_type',
                     'listing_exchange', 'conid', 'conidEx', 'open_close', 'directed_exchange', 'clearing_id',
                     'clearing_name', 'liquidation_trade', 'is_event_trading', 'order_ref', 'account_allocation_name']
        trades = self.__call_ibgateway('GET', '/iserver/account/trades')
        MTDB.save('IbTrade', trades, on_conflict='update', whitelist=whitelist)
        return pd.DataFrame(trades)

    def download_orders(self) -> pd.DataFrame | None:
        # the list may be incomplete, no definitive document is available
        whitelist = [
            'acct', 'exchange', 'conidex', 'conid', 'account', 'orderId', 'cashCcy', 'sizeAndFills', 'orderDesc',
            'description1', 'description2', 'ticker', 'secType', 'listingExchange', 'remainingQuantity',
            'filledQuantity', 'totalSize', 'companyName', 'status', 'order_ccp_status', 'avgPrice', 'origOrderType',
            'supportsTaxOpt', 'lastExecutionTime', 'orderType', 'bgColor', 'fgColor', 'order_ref', 'timeInForce',
            'lastExecutionTime_r', 'side', 'order_cancellation_by_system_reason', 'outsideRTH', 'price']
        orders = self.__call_ibgateway('GET', '/iserver/account/orders')
        if orders and orders['orders']:
            MTDB.save('IbOrder', orders['orders'], on_conflict='update', whitelist=whitelist)
            return pd.DataFrame(orders['orders'])

    def submit_order(self, plan: TradePlan, order: RawOrder) -> str | None:
        validator = InteractiveBrokersValidator(plan, self)
        validator.validate(order)
        ib_order = result = ex = broker_order_id = None
        try:
            ib_order = {
                'acctId': self.account_id,
                'conid': plan.broker_instrument_id(order.ticker),
                'cOID': order.id,
                'orderType': 'MOC' if plan.entry_type == 'TOC' else 'MKT',
                'listingExchange': 'SMART',
                'outsideRTH': False,
                'side': order.side.upper(),
                'ticker': order.ticker,
                'tif': 'DAY',
                'referrer': 'ParentOrder',
                'quantity': abs(order.size),
                'useAdaptive': False,
                'isClose': False
            }
            results = []
            result = self.__call_ibgateway(
                'POST', f'/iserver/account/{self.account_id}/orders', json={'orders': [ib_order]})
            results.append(result)
            print(f'Submit order response: {result}')
            # Sometimes IB needs additional confirmation before submitting order. Confirm yes to the message.
            # https://interactivebrokers.github.io/cpwebapi/endpoints
            while 'id' in result[0]:
                result = self.__call_ibgateway('POST', f'/iserver/reply/{result[0]["id"]}', json={'confirmed': True})
                results.append(result)
                print(f'Confirm order response: {result}')
            broker_order_id = result[0]['order_id']
            return broker_order_id
        except Exception as e:
            ex = traceback.format_exc()
            raise RuntimeError(f'Placing order failed for {self.account.alias} order {order}') from e
        finally:
            self.__log_order(self.account_id, plan, order, ib_order, results, ex, broker_order_id)

    def cancel_order(self, plan: TradePlan, order: RawOrder = None) -> bool:
        order_refs = [o.id for o in plan.get_orders()] if order is None else [order.id]
        broker_orders = self.download_orders()
        if broker_orders is not None and 'order_ref' in broker_orders.columns:
            broker_orders.set_index('orderId', drop=True, inplace=True)
            broker_orders = broker_orders[broker_orders['order_ref'].isin(order_refs)]
            for order_id, order in broker_orders.iterrows():
                if order['status'] in ['Cancelled', 'Filled']:
                    continue
                try:
                    result = self.__call_ibgateway(
                        'DELETE', f'/iserver/account/{self.account_id}/order/{order_id}')
                    print(f'Cancel order {order_id}: {result}')
                except Exception as e:
                    raise RuntimeError(f'Cancelling order failed for order {order_id}') from e

    def __log_order(self, account_id: str, plan: TradePlan, order: RawOrder, iborder: dict,
                    result: Any, exception: str, broker_order_id: str):
        '''Log the input and output of order submission to database.'''
        log = IbOrderLog(
            id=MTDB.uniqueid(),
            order_id=order.id,
            account_id=account_id,
            plan=plan,
            order=order,
            iborder=iborder,
            result=result,
            exception=exception,
            broker_order_id=broker_order_id,
            log_time=datetime.utcnow()
        )
        MTDB.save('IbOrderLog', log, on_conflict='error')

    @lru_cache(maxsize=100)
    def resolve_tickers(self, ticker_css) -> dict[str, list]:
        result = self.__call_ibgateway('GET', '/trsrv/stocks', {'symbols': ticker_css})
        for k, v in result.items():
            # sort ticker and put that of US market first
            options = sorted(v, key=lambda v: not v['contracts'][0]['isUS'])
            # return ticker to broker options mapping in the format like
            # {TK1: [{'id': id1, 'label': label1}, {'id': id2, 'label': label2}], TK2: [{'id': id1, 'label': label1}]}
            result[k] = [{
                'id': _['contracts'][0]['conid'],
                'label': f'{_["name"]} {str(_["contracts"][0])}'
            } for _ in options]
        return result

    def find_trades(self, order: RawOrder) -> list(dict) | None:
        return MTDB.get_all('IbTrade', 'order_ref', order.id, cls=dict)

    def find_order(self, order: RawOrder) -> dict | None:
        return MTDB.get_one('IbOrder', 'order_ref', order.id, cls=dict)

    def format_trades(self, orders: list[RawOrder]) -> list[dict]:
        trades = []
        for order in orders:
            for status in self.find_trades(order):
                # IB return trade time in UTC time zone, but it is not marked as UTC
                entry_time = datetime.strptime(status['trade_time'], '%Y%m%d-%H:%M:%S').replace(tzinfo=timezone.utc)
                trade = {'ticker': order.ticker,
                         'entry_time': entry_time,
                         'size': status['size'] if status['side'] == 'B' else -status['size'],
                         'entry_price': float(status['price']),
                         'commission': float(status['commission'])}
                trades.append(trade)
        return trades

    def daily_bar(self, conid: str, start: str, end: str = None) -> pd.DataFrame:
        period = (datetime.now().date() - datetime.strptime(start, '%Y-%m-%d').date()).days
        bars = self.__call_ibgateway(
            'GET', f'/iserver/marketdata/history?conid={conid}&period={period}d&bar=1d&outsideRth=0')
        df = pd.DataFrame(bars['data'])
        df.rename(columns={'o': 'Open', 'c': 'Close', 'h': 'High', 'l': 'Low', 'v': 'Volume'}, inplace=True)
        df.index = pd.to_datetime(df['t'], unit='ms', utc=True)
        df.index.rename('Date', inplace=True)
        df.index = df.index.tz_localize(None).normalize()
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df[start:end]
        return df

    def spot(self, conids: list[str]) -> pd.DataFrame:
        prices = self.__call_ibgateway('GET', f'/iserver/marketdata/snapshot?conids={",".join(str(conids))}')
        df = pd.DataFrame(prices)
        return df


class InteractiveBrokersValidator(OrderValidator):

    def __init__(self, plan: TradePlan, broker: Broker, pytest_now: datetime = None):
        super().__init__(plan)
        self.broker = broker
        self.pytest_now = pytest_now  # allow injecting fake current time for pytest
        self.tests.extend([
            self.order_type_is_supported,
            self.order_not_in_finished_trades,
            self.order_not_in_open_orders,
        ])

    def order_type_is_supported(self, order: RawOrder):
        plan = TradePlan.get_plan(order.plan_id)
        if plan.strict:
            # only support TOC/TOO orders in strict mode since others are not repeatable
            self._assert_is_in(plan.entry_type, ['TOO', 'TOC'], 'Order type is not supported in strict mode')

    def order_not_in_open_orders(self, order: RawOrder):
        broker_order = self.broker.find_order(order)
        # Okay if no order exists or order exists but not fully submitted
        self._assert(broker_order is None or broker_order['status'] == 'Inactive', 'Order exists in IbOrder table')

    def order_not_in_finished_trades(self, order: RawOrder):
        broker_trades = self.broker.find_trades(order)
        self._assert_equal(len(broker_trades), 0, 'Order exists in finished trade table')

    def order_size_is_within_limit(self, order: RawOrder):
        # TODO: should be configurable in plan
        self._assert_less_than(abs(order.size), 10000, 'Order size is too big')

    def order_in_time_window(self, order: RawOrder):
        # TODO: implement this
        pass
