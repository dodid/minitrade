from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import attrs
import pandas as pd
import requests

from minitrade.broker import Broker, BrokerAccount
from minitrade.trader import TradePlan
from minitrade.utils.config import config
from minitrade.utils.convert import attrs_to_str, iso_to_datetime, obj_to_str
from minitrade.utils.mtdb import MTDB

if TYPE_CHECKING:
    from minitrade.backtest import RawOrder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@attrs.define(slots=False, kw_only=True)
class IbTrade:
    execution_id: str
    symbol: str
    supports_tax_opt: str
    side: str
    order_description: str
    trade_time: str
    trade_time_r: int
    size: float
    price: str
    submitter: str
    exchange: str
    commission: str
    net_amount: float
    account: str
    accountCode: str
    company_name: str
    contract_description_1: str
    sec_type: str
    listing_exchange: str
    conid: int
    conidEx: str
    directed_exchange: str
    clearing_id: str
    clearing_name: str
    liquidation_trade: str
    is_event_trading: str
    order_ref: str

    @classmethod
    def from_row(cls, row):
        return cls(**row)


@attrs.define(slots=False, kw_only=True)
class IbOrder:
    acct: str
    conidex: str
    conid: int
    orderId: int
    cashCcy: str
    sizeAndFills: str
    orderDesc: str
    description1: str
    ticker: str
    secType: str
    listingExchange: str
    remainingQuantity: float
    filledQuantity: float
    companyName: str
    status: str
    outsideRTH: bool
    origOrderType: str
    supportsTaxOpt: str
    lastExecutionTime: str
    orderType: str
    bgColor: str
    fgColor: str
    price: str
    order_ref: str
    timeInForce: str
    lastExecutionTime_r: int
    side: str
    order_cancellation_by_system_reason: str

    @classmethod
    def from_row(cls, row):
        return cls(**row)


@attrs.define(slots=False, kw_only=True)
class IbOrderLog:
    id: str
    trace_id: str
    account_id: str
    plan: str = attrs.field(converter=attrs_to_str)
    raworder: str = attrs.field(converter=attrs_to_str)
    iborder: str = attrs.field(converter=obj_to_str)
    result: str = attrs.field(converter=obj_to_str)
    exception: str = None
    broker_order_id: str = None
    log_time: datetime = attrs.field(converter=iso_to_datetime)

    @classmethod
    def from_row(cls, row):
        return cls(**row)


class BrokerIB(Broker):

    def __init__(self, account: BrokerAccount):
        super().__init__(account)
        self._account_id = None
        self._admin_host = config.brokers.ib.gateway_admin_host
        self._admin_port = config.brokers.ib.gateway_admin_port
        self._port = None

    def __call_ibgateway_admin(self, method: str, path: str, params: dict | None = None):
        '''Call the ibgateway's admin API

        Parameters
        ----------
        method : str
            The HTTP method, i.e. GET, POST, PUT, DELETE, etc.
        path : str
            The REST API endpoint, e.g. '/ibgateway'
        params : dict, optional
            Extra parameters to be sent along with the REST API call

        Returns
        -------
        Any
            result json if API returns 200, otherwise None
        '''
        url = f'http://{self._admin_host}:{self._admin_port}{path}'
        resp = requests.request(method=method, url=url, params=params)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code >= 400:
            raise RuntimeError(f'Request {path} returned {resp.status_code} {resp.text}')

    def __call_ibgateway(
            self, method: str, path: str, params: dict | None = None, json: Any | None = None, timeout: int = 10) -> Any:
        '''Call the ibgateway's REST API

        Parameters
        ----------
        method : str
            The HTTP method, i.e. GET, POST, PUT, DELETE, etc.
        path : str
            The REST API endpoint, see https://www.interactivebrokers.com/api/doc.html
        params : dict, optional
            Extra parameters to be sent along with the REST API call

        Returns
        -------
        Any
            result json if API returns 200, otherwise None
        '''
        if self._port:
            url = f'http://localhost:{self._port}/v1/api{path}'
            resp = requests.request(method=method, url=url, params=params, json=json, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code >= 400:
                raise RuntimeError(f'Request {path} returned {resp.status_code} {resp.text}')
        else:
            raise RuntimeError(f'IB gateway port is not set for request {method} {path}')

    def is_broker_ready(self) -> bool:
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

    def __ib_login(self) -> bool:
        try:
            assert self.account.broker == 'IB'
            self.__call_ibgateway_admin('PUT', f'/ibgateway/{self.account.alias}')
            time.sleep(5)   # wait a few seconds for gateway to be authenticated
            return self.is_broker_ready()
        except Exception:
            logger.exception(f'Login failed for IB account {self.account.alias}')
            return False

    def __ib_logout(self) -> None:
        try:
            self.__call_ibgateway_admin('DELETE', f'/ibgateway/{self.account.alias}')
            self._port = None
        except Exception:
            logger.exception(f'Logout failed for IB account {self.account.alias}')

    def prepare(self) -> bool:
        '''Make broker account ready to receive orders

        Parameters
        ----------
        alias : str
            The broker alias

        Returns
        -------
        ready
            True if broker account is ready, otherwise False
        '''
        if self.is_broker_ready():
            return True
        else:
            return self.__ib_login()

    def release(self) -> None:
        self.__ib_logout()

    def get_metainfo(self) -> Any:
        try:
            accounts = self.__call_ibgateway('GET', '/portfolio/accounts')
            if accounts:
                self._account_id = accounts[0]['id']
            return accounts
        except Exception as e:
            raise RuntimeError(f'Getting account info failed for {self.account.alias}') from e

    def get_portfolio(self) -> pd.DataFrame:
        try:
            if not self._account_id:
                self.get_metainfo()
            portfolio = []
            i = 0
            while True:
                page = self.__call_ibgateway('GET', f'/portfolio/{self._account_id}/positions/{i}')
                portfolio.extend(page)
                if len(page) < 100:
                    break
                else:
                    i += 1
            return pd.DataFrame(portfolio)
        except Exception as e:
            raise RuntimeError(
                f'Getting account portfolio failed for {self.account.alias} ID {self._account_id}') from e

    def download_trades(self) -> pd.DataFrame:
        whitelist = [
            'execution_id', 'symbol', 'supports_tax_opt', 'side', 'order_description', 'trade_time', 'trade_time_r',
            'size', 'price', 'submitter', 'exchange', 'commission', 'net_amount', 'account', 'accountCode',
            'company_name', 'contract_description_1', 'sec_type', 'listing_exchange', 'conid', 'conidEx',
            'directed_exchange', 'clearing_id', 'clearing_name', 'liquidation_trade', 'is_event_trading', 'order_ref']
        try:
            trades = self.__call_ibgateway('GET', '/iserver/account/trades')
            MTDB.save_objects(trades, 'ibtrade', on_conflict='update', whitelist=whitelist)
            return pd.DataFrame(trades)
        except Exception as e:
            raise RuntimeError(f'Getting account trades failed for {self.account.alias}') from e

    def download_orders(self) -> pd.DataFrame | None:
        # the list may be incomplete, no definitive document is available
        whitelist = [
            'acct', 'conidex', 'conid', 'orderId', 'cashCcy', 'sizeAndFills', 'orderDesc', 'description1', 'ticker',
            'secType', 'listingExchange', 'remainingQuantity', 'filledQuantity', 'companyName', 'status',
            'origOrderType', 'supportsTaxOpt', 'lastExecutionTime', 'orderType', 'bgColor', 'fgColor', 'order_ref',
            'timeInForce', 'lastExecutionTime_r', 'side', 'order_cancellation_by_system_reason', 'outsideRTH', 'price']
        try:
            orders = self.__call_ibgateway('GET', '/iserver/account/orders')
            if orders and orders['orders']:
                MTDB.save_objects(orders['orders'], 'iborder', on_conflict='update', whitelist=whitelist)
                return pd.DataFrame(orders['orders'])
        except Exception as e:
            raise RuntimeError(f'Getting account orders failed for {self.account.alias}') from e

    def place_order(self, plan: TradePlan, order: RawOrder, trace_id: str = None) -> str | None:
        '''Place an order via IB

        Parameters
        ----------
        plan: TradePlan
            The trade plan
        orders : RawOrder
            Orders to be placed
        trace_id: str
            ID to trace order placement in workflow

        Returns
        -------
        order_id
            Order ID if order is placed, otherwise None
        '''
        ib_order = result = ex = broker_order_id = None
        try:
            if not self._account_id:
                self.get_metainfo()
            ib_order = {
                'acctId': self._account_id,
                'conid': plan.broker_ticker(order.ticker),
                'cOID': order.id,
                'orderType': 'MKT',
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
            result = self.__call_ibgateway(
                'POST', f'/iserver/account/{self._account_id}/orders', json={'orders': [ib_order]})
            # Sometimes IB needs additional confirmation before submitting order. Confirm yes to the message.
            # https://interactivebrokers.github.io/cpwebapi/endpoints
            if 'id' in result[0]:
                result = self.__call_ibgateway('POST', f'/iserver/reply/{result[0]["id"]}', json={'confirmed': True})
            broker_order_id = result[0]['order_id']
            return broker_order_id
        except Exception as e:
            ex = traceback.format_exc()
            raise RuntimeError(f'Placing order failed for {self.account.alias} order {order}') from e
        finally:
            self.__log_order(trace_id, self._account_id, plan, order, ib_order, result, ex, broker_order_id)

    def __log_order(self, trace_id: str, account_id: str, plan: TradePlan, raworder: RawOrder, iborder: dict,
                    result: Any, exception: str, broker_order_id: str):
        '''Log the input and output of order submission to database.'''
        log = IbOrderLog(
            id=MTDB.uniqueid(),
            trace_id=trace_id,
            account_id=account_id,
            plan=plan,
            raworder=raworder,
            iborder=iborder,
            result=result,
            exception=exception,
            broker_order_id=broker_order_id,
            log_time=datetime.utcnow()
        )
        MTDB.save_objects(log, 'iborderlog', on_conflict='error')

    def resolve_tickers(self, ticker_css) -> dict[str, list]:
        try:
            ticker_css = ticker_css.replace(' ', '')
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
        except Exception as e:
            raise RuntimeError(f'Resolving tickers failed for {self.account.alias} ticker {ticker_css}') from e

    def get_cached_trade_status(self, order: RawOrder) -> list(dict) | None:
        try:
            rows = MTDB.get_objects('ibtrade', 'order_ref', order.id, dict)
            return rows
        except Exception as e:
            raise RuntimeError(f'Looking up trade status failed for order {order}') from e

    def get_cached_order_status(self, order: RawOrder) -> dict | None:
        try:
            row = MTDB.get_object('iborder', 'order_ref', order.id)
            return dict(row) if row is not None else None
        except Exception as e:
            raise RuntimeError(f'Looking up order status failed for order {order}') from e

    def get_cached_trades(self, orders: list[RawOrder]) -> list[dict]:
        trades = []
        for order in orders:
            for status in self.get_cached_trade_status(order):
                entry_time = datetime.strptime(status['trade_time'], '%Y%m%d-%H:%M:%S').replace(tzinfo=timezone.utc)
                trade = {'ticker': order.ticker,
                         'entry_time': entry_time,
                         'size': status['size'] if status['side'] == 'B' else -status['size'],
                         'entry_price': float(status['price']),
                         'commission': float(status['commission'])}
                trades.append(trade)
        return trades
