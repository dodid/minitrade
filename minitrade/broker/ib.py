from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
            self, method: str, path: str, params: dict | None = None, json: Any | None = None, timeout: int = 10) -> Any:
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
        try:
            status = self.__call_ibgateway_admin('GET', f'/ibgateway/{self.account.alias}')
            if status and status['account'] == self.account.username:
                self._port = status['port']
                if status['authenticated']:
                    return True
                else:
                    # if not authenticated, try reauthenticate
                    self.__call_ibgateway('POST', '/iserver/reauthenticate')
                    # wait for reauthentication to finish
                    time.sleep(5)
                    # try getting status again
                    status = self.__call_ibgateway_admin('GET', f'/ibgateway/{self.account.alias}')
                    if status:
                        self._port = status['port']
                    # return the renewed status as is
                    return status and self.account.username == status['account'] and status['authenticated']
        except Exception:
            logger.exception(f'Checking broker status failed for {self.account.alias}')
        return False

    def connect(self):
        if not self.is_ready():
            try:
                self.__call_ibgateway_admin('PUT', f'/ibgateway/{self.account.alias}')
                time.sleep(5)   # wait a few seconds for gateway to be authenticated
                if not self.is_ready():
                    raise ConnectionError('Login failed for {self.account.alias}')
            except Exception as e:
                raise ConnectionError(f'Login failed for {self.account.alias}') from e
        self.get_account_info()

    def disconnect(self):
        try:
            self.__call_ibgateway_admin('DELETE', f'/ibgateway/{self.account.alias}')
            self._port = None
        except Exception:
            logger.exception(f'Logout failed for IB account {self.account.alias}')

    def get_account_info(self) -> Any:
        return self.__call_ibgateway('GET', '/portfolio/accounts')

    def get_portfolio(self) -> pd.DataFrame:
        portfolio = []
        i = 0
        while True:
            page = self.__call_ibgateway('GET', f'/portfolio/{self.account_id}/positions/{i}')
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
                     'clearing_name', 'liquidation_trade', 'is_event_trading', 'order_ref']
        trades = self.__call_ibgateway('GET', '/iserver/account/trades')
        MTDB.save(trades, 'IbTrade', on_conflict='update', whitelist=whitelist)
        return pd.DataFrame(trades)

    def download_orders(self) -> pd.DataFrame | None:
        # the list may be incomplete, no definitive document is available
        whitelist = ['acct', 'exchange', 'conidex', 'conid', 'orderId', 'cashCcy', 'sizeAndFills', 'orderDesc',
                     'description1', 'description2', 'ticker', 'secType', 'listingExchange', 'remainingQuantity',
                     'filledQuantity', 'companyName', 'status', 'order_ccp_status', 'origOrderType', 'supportsTaxOpt',
                     'lastExecutionTime', 'orderType', 'bgColor', 'fgColor', 'order_ref', 'timeInForce',
                     'lastExecutionTime_r', 'side', 'order_cancellation_by_system_reason', 'outsideRTH', 'price']
        orders = self.__call_ibgateway('GET', '/iserver/account/orders')
        if orders and orders['orders']:
            MTDB.save(orders['orders'], 'IbOrder', on_conflict='update', whitelist=whitelist)
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
                'orderType': 'MOC' if order.entry_type == 'TOC' else 'MKT',
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
            logger.info(f'Submit order response: {result}')
            # Sometimes IB needs additional confirmation before submitting order. Confirm yes to the message.
            # https://interactivebrokers.github.io/cpwebapi/endpoints
            if 'id' in result[0]:
                result = self.__call_ibgateway('POST', f'/iserver/reply/{result[0]["id"]}', json={'confirmed': True})
                results.append(result)
                logger.info(f'Confirm order response: {result}')
            broker_order_id = result[0]['order_id']
            return broker_order_id
        except Exception as e:
            ex = traceback.format_exc()
            raise RuntimeError(f'Placing order failed for {self.account.alias} order {order}') from e
        finally:
            self.__log_order(self.account_id, plan, order, ib_order, results, ex, broker_order_id)

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
        MTDB.save(log, 'IbOrderLog', on_conflict='error')

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


class InteractiveBrokersValidator(OrderValidator):

    def __init__(self, plan: TradePlan, broker: Broker, pytest_now: datetime = None):
        super().__init__(plan)
        self.broker = broker
        self.pytest_now = pytest_now  # allow injecting fake current time for pytest
        self.tests.extend([
            self.order_size_is_within_limit,
            self.order_in_time_window,
            self.order_not_in_finished_trades,
            self.order_not_in_open_orders,
        ])

    def order_size_is_within_limit(self, order: RawOrder):
        self._assert_less_than(abs(order.size), 10000, 'Order size is too big')

    def order_in_time_window(self, order: RawOrder):
        plan = TradePlan.get_plan(order.plan_id)
        now = self.pytest_now or datetime.now(tz=ZoneInfo(plan.market_timezone))
        usmarket = MTDB.get_one('NasdaqTraded', 'symbol', order.ticker) is not None
        self._assert_equal(usmarket, True, 'Only U.S. market is supported for now')
        market_open, market_close = timedelta(hours=9, minutes=30), timedelta(hours=16)
        if order.entry_type == 'TOO':
            # TOO order submit window is between market close on signal_time date and
            # before next market open, considering weekends but not holidays
            self._assert_less_than(order.signal_time + market_close, now,
                                   'TOO order must be submitted after market close')
            self._assert_less_than(
                now, order.signal_time + market_open + timedelta(days=1 if order.signal_time.weekday() < 4 else 3),
                'TOO order must be submitted before next market open')
        elif order.entry_type == 'TOC':
            # TOC order submit window is before market close on signal_time date
            self._assert_less_than(now, order.signal_time + market_close,
                                   'TOC order must be submitted before market close')
        else:
            self._assert(False, f'Unknown order entry type: {order.entry_type}')

    def order_not_in_open_orders(self, order: RawOrder):
        broker_order = self.broker.find_order(order)
        # Okay if no order exists or order exists but not fully submitted
        self._assert(broker_order is None or broker_order['status'] == 'Inactive', 'Order exists in IbOrder table')

    def order_not_in_finished_trades(self, order: RawOrder):
        broker_trades = self.broker.find_trades(order)
        self._assert_equal(len(broker_trades), 0, 'Order exists in finished trade table')
