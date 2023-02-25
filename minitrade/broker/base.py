from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from minitrade.utils.mtdb import MTDB

if typing.TYPE_CHECKING:
    from minitrade.backtest import RawOrder
    from minitrade.trader import TradePlan

__all__ = [
    'BrokerAccount',
    'Broker',
]


@dataclass(kw_only=True)
class BrokerAccount:
    alias: str
    broker: str
    mode: str
    username: str
    password: str

    @staticmethod
    def get_account(plan: TradePlan | str) -> BrokerAccount:
        '''Look up a broker account from a trade plan

        Parameters
        ----------
        plan : TradePlan | str
            Trade plan or a trade plan alias

        Returns
        -------
        account
            Broker account if found, otherwise None
        '''
        alias = plan if isinstance(plan, str) else plan.broker_account
        return MTDB.get_one('brokeraccount', 'alias', alias, cls=BrokerAccount)

    @staticmethod
    def list() -> list[BrokerAccount]:
        '''Return the list of available broker accounts

        Returns
        -------
        list
            A list of broker accounts
        '''
        return MTDB.get_all('brokeraccount', cls=BrokerAccount, orderby='alias')

    def save(self) -> None:
        '''Save broker account to database. 
        '''
        MTDB.save(self, 'brokeraccount', on_conflict='error')

    def delete(self) -> None:
        '''Delete broker account from database. 

        Raises
        ------
        RuntimeError
            If deleting broker account failed
        '''
        MTDB.delete('brokeraccount', 'alias', self.alias)


class Broker(ABC):

    AVAILABLE_BROKERS = {'IB': 'Interactive Brokers'}

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
            from .ib import InteractiveBrokers
            return InteractiveBrokers(account)
        else:
            raise AttributeError(f'Broker {account.broker} is not supported')

    def __init__(self, account: BrokerAccount):
        ''' Create a broker connection with the supplied `account` info. 

        Parameters
        ----------
        account : BrokerAccount
            Account info
        '''
        self.account = account

    @abstractmethod
    def is_ready(self) -> bool:
        '''Return True if the broker account is connected and ready to receive orders

        Returns
        -------
        ready
            True if ready, otherwise False
        '''
        raise NotImplementedError()

    @abstractmethod
    def connect(self):
        '''Set up a working connection to the broker account.

        Raises
        -------
        ConnectionError
            Raise ConnectionError if a working connection can't be established.
        '''
        raise NotImplementedError()

    @abstractmethod
    def submit_order(self, plan: TradePlan, order: RawOrder, trace_id: str = None) -> str:
        '''Submit an order via broker interface

        Parameters
        ----------
        plan: TradePlan
            The trade plan
        order : RawOrder
            Order to be placed
        trace_id: str
            ID to trace order as part of a workflow

        Returns
        -------
        order_id
            Broker assigned order ID if order is submitted successfully, otherwise None
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_account_info(self) -> dict:
        '''Get broker account meta info

        Returns
        -------
        info
            A dict that contains broker specific account information
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_portfolio(self) -> pd.DataFrame:
        '''Get account portfolio

        Returns
        -------
        portfolio
            A dataframe that contains broker specific portfolio infomation
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
        '''
        raise NotImplementedError()

    @abstractmethod
    def disconnect(self) -> None:
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
        '''
        raise NotImplementedError()

    @abstractmethod
    def find_trades(self, order: RawOrder) -> dict:
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
        '''
        raise NotImplementedError()

    @abstractmethod
    def find_order(self, order: RawOrder) -> dict:
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
        '''
        raise NotImplementedError()

    @abstractmethod
    def format_trades(self, orders: list[RawOrder]) -> list[dict]:
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
        '''
        raise NotImplementedError()
