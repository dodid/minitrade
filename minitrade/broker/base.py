from __future__ import annotations

import logging
import typing
from abc import ABC, abstractmethod

import attrs
import pandas as pd
from pypika import Query, Table

from minitrade.utils.config import config
from minitrade.utils.mtdb import MTDB

if typing.TYPE_CHECKING:
    from minitrade.backtest import RawOrder
    from minitrade.trader import TradePlan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@attrs.define(slots=False, kw_only=True)
class BrokerAccount:
    alias: str
    broker: str
    mode: str
    username: str
    password: str

    @staticmethod
    def get_account(plan_or_alias: TradePlan | str) -> BrokerAccount:
        '''Look up a broker account from a trade plan

        Parameters
        ----------
        plan_or_alias : TradePlan | str
            Trade plan that contains broker account alias or the alias itself

        Returns
        -------
        account
            Broker account if found

        Raises
        ------
        RuntimeError
            If looking up broker account failed
        '''
        alias = plan_or_alias if isinstance(plan_or_alias, str) else plan_or_alias.broker_account
        if alias:
            try:
                brokeraccount = Table('brokeraccount')
                stmt = Query.from_(brokeraccount).select('*').where(brokeraccount['alias'] == alias)
                with MTDB.conn() as conn:
                    row = conn.execute(str(stmt)).fetchone()
                return BrokerAccount.from_row(row)
            except Exception as e:
                raise RuntimeError(f'Looking up broker account {alias} failed') from e
        raise RuntimeError(f'Cannot find broker account {alias}')

    @staticmethod
    def list_accounts() -> list[BrokerAccount]:
        '''Return the list of available broker accounts

        Returns
        -------
        list
            A list of broker accounts

        Raises
        ------
        RuntimeError
            If looking up broker account failed
        '''
        try:
            with MTDB.conn() as conn:
                rows = conn.execute('SELECT * FROM brokeraccount ORDER BY alias').fetchall()
            return [BrokerAccount.from_row(row) for row in rows]
        except Exception as e:
            raise RuntimeError('Looking up broker account failed') from e

    @classmethod
    def from_row(cls, row):
        return cls(**row)

    def save(self) -> None:
        '''Save broker account to database. 

        Overwrite existing ones if account alias already exists.

        Raises
        ------
        RuntimeError
            If saving broker account failed
        '''
        try:
            data = attrs.asdict(self)
            brokeraccount = Table('brokeraccount')
            stmt = Query.into(brokeraccount).columns(*data.keys()).insert(*data.values())
            with MTDB.conn() as conn:
                conn.execute(str(stmt))
        except Exception as e:
            raise RuntimeError(f'Saving broker account {self.alias} failed') from e

    def delete(self) -> None:
        '''Delete broker account from database. 

        Raises
        ------
        RuntimeError
            If deleting broker account failed
        '''
        try:
            brokeraccount = Table('brokeraccount')
            stmt = Query.from_(brokeraccount).delete().where(brokeraccount['alias'] == self.alias)
            with MTDB.conn() as conn:
                conn.execute(str(stmt))
        except Exception as e:
            raise RuntimeError(f'Deleting broker account {self.alias} failed') from e


class Broker(ABC):

    @staticmethod
    def get_supported_brokers() -> dict[str, str]:
        ''' Return supported brokers

        Returns
        -------
        dict
            A dict that maps the short broker name to a long friendly name for supported brokers
        '''
        return {'IB': 'Interactive Brokers'}

    @staticmethod
    def get_broker(account: BrokerAccount) -> Broker:
        ''' Create a broker connection for the specific `account`

        Parameters
        ----------
        account : BrokerAccount
            Broker account that contains broker type and credential

        Returns
        -------
        broker
            Broker connection instance

        Raises
        ------
        NotImplementedError
            If the broker is not supported
        '''
        if account.broker == 'IB':
            from .ib import BrokerIB
            return BrokerIB(account)
        else:
            raise NotImplementedError(f'Broker {account.broker} is not supported')

    def __init__(self, account: BrokerAccount):
        ''' Create a broker connection with the supplied `account` info. 

        Parameters
        ----------
        account : BrokerAccount
            Account info
        '''
        self.account = account

    @abstractmethod
    def is_broker_ready(self) -> bool:
        '''Return if the broker account is connected and ready to receive orders

        Returns
        -------
        ready
            True if ready, otherwise False
        '''
        raise NotImplementedError()

    @abstractmethod
    def prepare(self, alias: str) -> bool:
        '''Set up a working connection to the broker account identified by `alias`.

        Parameters
        ----------
        alias : str
            Account alias

        Returns
        -------
        ready
            True if ready, otherwise False
        '''
        raise NotImplementedError()

    @abstractmethod
    def place_order(self, plan: TradePlan, order: RawOrder, trace_id: str = None) -> str:
        '''Place an order 

        Parameters
        ----------
        plan : TradePlan
            The trade plan that contains broker specific ticker mapping
        order : RawOrder
            The raw order generated by backtesting

        Returns
        -------
        order_id
            Broker order ID

        Raises
        ------
        RuntimeError
            If placing order fails for any reason
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_metainfo(self) -> dict:
        '''Get broker account meta info

        Returns
        -------
        info
            A dict that contains broker specific account information

        Raises
        ------
        RuntimeError
            If getting infomation fails for any reason
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_portfolio(self) -> pd.DataFrame:
        '''Get account portfolio

        Returns
        -------
        portfolio
            A dataframe that contains broker specific portfolio infomation

        Raises
        ------
        RuntimeError
            If getting portfolio fails for any reason
        '''
        raise NotImplementedError()

    @abstractmethod
    def download_trades(self) -> pd.DataFrame:
        '''Get recent trades

        The number of trades returned depends on the data availability of each broker.

        Returns
        -------
        trades
            A dataframe that contains recently finished trades

        Raises
        ------
        RuntimeError
            If getting trades fails for any reason
        '''
        raise NotImplementedError()

    @abstractmethod
    def download_orders(self) -> pd.DataFrame:
        '''Get recent orders

        The number of orders returned depends on the data availability of each broker.

        Returns
        -------
        orders
            A dataframe that contains recent orders which can be submitted, filled, or cancelled

        Raises
        ------
        RuntimeError
            If getting order fails for any reason
        '''
        raise NotImplementedError()

    @abstractmethod
    def release(self) -> None:
        '''Disconnect from broker and release any resources.
        '''
        raise NotImplementedError()

    @abstractmethod
    def resolve_tickers(self, ticker_css: str) -> dict[str, list]:
        '''Resolve generic tickers to broker specific ticker IDs.

        Parameters
        ----------
        ticker_css : str
            A list of tickers formatted as a comma separated string

        Returns
        -------
        map
            A dict mapping each generic ticker name to a list of broker ticker options.
            The format is like 
            {
                TK1: [
                    {'id': id1, 'label': label1}, 
                    {'id': id2, 'label': label2}
                ], 
                TK2: [
                    {'id': id1, 'label': label1}
                ],
                TK3: []
            }
            where id is the broker specific ticker ID, label is a human readable string
            to help disambiguate the options.

        Raises
        ------
        RuntimeError
            If resolving tickers fails for any reason
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_cached_trade_status(self, order: RawOrder) -> dict:
        '''Get trade information associated with the raw order.

        Parameters
        ----------
        order : RawOrder
            A raw order

        Returns
        -------
        trade
            A dict that contains broker specific trade infomation associated with the order, 
            or None if no trade is found or 

        Raises
        ------
        RuntimeError
            If getting trade status fails for any reason
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_cached_order_status(self, order: RawOrder) -> dict:
        '''Get last known order status, not necessarily latest.

        Parameters
        ----------
        order : RawOrder
            A raw order

        Returns
        -------
        order
            A dict that contains broker specific order infomation associated with the order, 
            or None if no broker order is found or 

        Raises
        ------
        RuntimeError
            If getting order status fails for any reason
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_cached_trades(self, orders: list[RawOrder]) -> list[dict]:
        '''Get trade status corresponding to the list of orders and format as the following
            {
                'ticker': ...,
                'entry_time': ...,
                'size': ...,
                'entry_price': ...,
                'commission': ...
            }

        Parameters
        ----------
        order : list[RawOrder]
            A list of raw orders

        Returns
        -------
        trades
            A list of completed trades with relevant information

        Raises
        ------
        RuntimeError
            If getting trade status fails for any reason
        '''
        raise NotImplementedError()
