from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from minitrade.utils.mtdb import MTDB

if typing.TYPE_CHECKING:
    from minitrade.trader import RawOrder, TradePlan

__all__ = [
    'BrokerAccount',
    'Broker',
]


@dataclass(kw_only=True)
class BrokerAccount:
    '''
    BrokerAccount describes a broker account, such as name, type, and login credentials.
    '''

    alias: str
    '''Account alias'''

    broker: str
    '''Broker alias as in `Broker.AVAILABLE_BROKERS`'''

    mode: str
    '''"Live" or "Paper". Note this is only a hint to help remember. Whether an account
    is live or paper depends on the account itself rather than this field.'''

    username: str
    '''Account username'''

    password: str
    '''Account password'''

    @staticmethod
    def get_account(plan_or_alias: TradePlan | str) -> BrokerAccount:
        '''Look up a broker account from a broker account `alias` or the alias 
        as specified in a `TradePlan`.

        Args:
            plan_or_alias: A trade plan or a broker account alias

        Returns:
            A broker account, or None if the alias is invalid
        '''
        alias = plan_or_alias if isinstance(plan_or_alias, str) else plan_or_alias.broker_account
        return MTDB.get_one('BrokerAccount', 'alias', alias, cls=BrokerAccount)

    @staticmethod
    def list() -> list[BrokerAccount]:
        '''Return the list of available broker accounts.

        Returns:
            A list of zero or more broker accounts
        '''
        return MTDB.get_all('BrokerAccount', cls=BrokerAccount, orderby='alias')

    def save(self) -> None:
        '''Save a broker account to database.'''
        MTDB.save(self, 'BrokerAccount', on_conflict='error')

    def delete(self) -> None:
        '''Delete broker account from database.'''
        MTDB.delete('BrokerAccount', 'alias', self.alias)


class Broker(ABC):
    '''
    Broker is a base class that manages the connection to a broker system. Extend this class 
    to add a concrete implementation to talk to a particular broker.
    '''

    AVAILABLE_BROKERS = {'IB': 'Interactive Brokers'}
    '''A dict mapping from broker alias to broker user-friendly name for supported brokers.'''

    @staticmethod
    def get_broker(account: BrokerAccount) -> Broker:
        '''Create a broker connector for a `account` that specifies the type of broker
        and the login credentials.

        Args:
            account: A `BrokerAccount` instance

        Returns: 
            A `Broker` instance

        Raises:
            AttributeError: If the type of broker is not supported
        '''
        if account.broker == 'IB':
            from .ib import InteractiveBrokers
            return InteractiveBrokers(account)
        else:
            raise AttributeError(f'Broker {account.broker} is not supported')

    def __init__(self, account: BrokerAccount):
        ''' Create a broker connector with the supplied `account` info. 

        Args:
            account: a `BrokerAccount` instance
        '''
        self.account = account

    @abstractmethod
    def is_ready(self) -> bool:
        '''Check if the broker connector has a working connection with the broker.

        All calls interacting with the broker system should only be sent after
        `is_ready` return True. 

        Returns: 
            True if a connection to broker is ready, otherwise False
        '''
        raise NotImplementedError()

    @abstractmethod
    def connect(self):
        '''Set up a working connection to the broker.

        Raises:
            ConnectionError: If a working connection can't be established.
        '''
        raise NotImplementedError()

    @abstractmethod
    def submit_order(self, plan: TradePlan, order: RawOrder) -> str:
        '''Submit an order to the broker.

        Args:
            plan: A trade plan
            order: A `RawOrder` to be submitted

        Returns:
            Broker assigned order ID if order is submitted successfully, otherwise None
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_account_info(self) -> dict:
        '''Get broker account meta info

        Returns:
            A dict that contains broker specific account information
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_portfolio(self) -> pd.DataFrame:
        '''Get account portfolio

        Returns:
            A dataframe that contains broker specific portfolio infomation
        '''
        raise NotImplementedError()

    @abstractmethod
    def download_trades(self) -> pd.DataFrame:
        '''Get recent trades at broker. Trades are completed orders.

        This function is called immediately before and after order submission to get the most 
        recent trade information on the broker side. The trades returned should be persisted
        to database and can be used for display, order validation, statistics, and record keeping.
        Trade info is usually final.

        The number of trades returned depends on the data availability of particular broker.

        Returns:
            A dataframe that contains recently finished trades
        '''
        raise NotImplementedError()

    @abstractmethod
    def download_orders(self) -> pd.DataFrame:
        '''Get recent orders at broker. RawOrder submitted creates a correspoing order
        on the broker side, which can have different status such as open, cancelled, error, etc.

        This function is called immediately before and after order submission to get the most 
        recent order information on the broker side. The orders returned should be persisted
        to database and can be used for display, order validation, statistics, and record keeping.
        Since order status can change over time, new status should overwrite past one.

        The number of orders returned depends on the data availability of each broker.

        Returns:
            A dataframe that contains recent order information
        '''
        raise NotImplementedError()

    @abstractmethod
    def disconnect(self) -> None:
        '''Disconnect from broker and release any resources.'''
        raise NotImplementedError()

    @abstractmethod
    def resolve_tickers(self, ticker_css: str) -> dict[str, list]:
        '''Resolve generic tickers to broker specific ticker IDs.

        Args:
            ticker_css: A list of tickers formatted as a comma separated string

        Returns:
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
        '''Get trade information associated with the raw `order`.

        A 'RawOrder' can be filled in multiple trades at different time and prices, therefore
        a list of trades may be returned.

        A subclass implementing this function only needs to look up trades from the database,
        instead of querying the broker directly, which can be slow. In scenarios where stale
        information can't be tolerated, `download_trades()` is always called before this 
        function is invoked.

        Args:
            order: A `RawOrder`

        Returns:
            A dict that contains broker specific trade infomation associated with the order, 
            or None if no broker trade is found.
        '''
        raise NotImplementedError()

    @abstractmethod
    def find_order(self, order: RawOrder) -> dict:
        '''Get last known order status, not necessarily latest.

        A subclass implementing this function only needs to look up orders from the database,
        instead of querying the broker directly, which can be slow. In scenarios where stale
        information can't be tolerated, `download_orders()` is always called before this 
        function is invoked.

        Args:
            order: A `RawOrder`

        Returns:
            A dict that contains broker specific order infomation associated with the order, 
            or None if no broker order is found.
        '''
        raise NotImplementedError()

    @abstractmethod
    def format_trades(self, orders: list[RawOrder]) -> list[dict]:
        '''Get broker specific trade status corresponding to the list of orders and format
            in dict like:
            {
                'ticker': ...,
                'entry_time': ...,
                'size': ...,
                'entry_price': ...,
                'commission': ...
            }

        Args:
            orders: A list of `RawOrder`

        Returns:
            A list of completed trades with relevant information
        '''
        raise NotImplementedError()
