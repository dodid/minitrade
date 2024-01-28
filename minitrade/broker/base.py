from __future__ import annotations

import traceback
import typing
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

from minitrade.utils.mtdb import MTDB

if typing.TYPE_CHECKING:
    from minitrade.trader import RawOrder, TradePlan

__all__ = [
    'BrokerAccount',
    'Broker',
    'OrderValidator',
    'OrderValidatorLog',
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
        MTDB.save('BrokerAccount', self, on_conflict='error')

    def delete(self) -> None:
        '''Delete broker account from database.'''
        MTDB.delete('BrokerAccount', 'alias', self.alias)


@dataclass(kw_only=True)
class OrderValidatorLog:
    '''OrderValidatorLog logs the results of order validation.'''

    id: str
    '''Validator run ID'''

    order_id: str
    '''Raw order ID'''

    order: str
    '''Raw order in Json format'''

    result: str
    '''Validation output'''

    exception: str | None = None
    '''Exception that occurs during validation'''

    log_time: datetime
    '''Log time'''


class OrderValidator:
    '''OrderValidator is responsible for integrity checks on a raw order before it's
    submitted to broker.'''

    def __init__(self, plan: TradePlan):
        self.plan = plan
        self.tests = [
            self.order_is_in_sync_with_db,
            self.order_has_correct_plan_id,
            self.order_has_correct_run_id,
            self.order_has_correct_ticker,
            self.order_has_correct_size,
            self.order_has_no_broker_order_id,
        ]

    def _assert(self, condition, message):
        if condition != True:
            raise AttributeError(message)

    def _assert_is_null(self, value, message):
        return self._assert(value is None, f'{message}: {value=}')

    def _assert_not_null(self, value, message):
        return self._assert(value is not None, f'{message}: {value=}')

    def _assert_equal(self, value, expected_value, message):
        return self._assert(value == expected_value, f'{message}: {value=} != {expected_value=}')

    def _assert_is_in(self, value, collection, message):
        return self._assert(value in collection, f'{message}: {value=} not in {collection=}')

    def _assert_less_than(self, value, limit, message):
        return self._assert(value < limit, f'{message}: {value=} >= {limit=}')

    def order_is_in_sync_with_db(self, order: RawOrder):
        order_in_db = MTDB.get_one('RawOrder', 'id', order.id, cls=dict)
        self._assert_not_null(order_in_db, 'Order is not in RawOrder table')
        self._assert_equal(asdict(order), order_in_db, 'Order is not in sync with db')

    def order_has_correct_plan_id(self, order: RawOrder):
        self._assert_equal(order.plan_id, self.plan.id, 'Order has incorrect plan ID')

    def order_has_correct_run_id(self, order: RawOrder):
        self._assert_not_null(MTDB.get_one('BacktestLog', 'id', order.run_id), 'Order has incorrect run ID')

    def order_has_correct_ticker(self, order: RawOrder):
        self._assert_is_in(order.ticker, self.plan.ticker_css.split(','), 'Order has incorrect ticker')
        self._assert_not_null(self.plan.broker_instrument_id(order.ticker), 'Order has incorrect ticker mapping')

    def order_has_correct_size(self, order: RawOrder):
        self._assert((order.size > 0 and order.side == 'Buy') or (
            order.size < 0 and order.side == 'Sell'), 'Order side and size does not agree')

    def order_has_no_broker_order_id(self, order: RawOrder):
        self._assert_is_null(order.broker_order_id, 'Order already has broker_order_id')

    def validate(self, order: RawOrder):
        '''Run a bunch of checks to ensure the raw order is valid.

        For example, a valid order should:

        - have valid values for all its attributes
        - not be a duplicate of what has been submitted before
        - be timely
        '''
        result, ex = {}, None
        try:
            for test in self.tests:
                test(order)
                result[test.__name__] = 'Pass'
        except Exception as e:
            result[test.__name__] = 'Fail'
            ex = traceback.format_exc()
            raise RuntimeError(f'Order {order.id} failed {test.__name__}') from e
        finally:
            self._log_validator_run(order, result, ex)

    def _log_validator_run(self, order: RawOrder, result: dict, exception: str) -> None:
        log = OrderValidatorLog(
            id=MTDB.uniqueid(),
            order_id=order.id,
            order=order,
            result=result,
            exception=exception,
            log_time=datetime.utcnow()
        )
        MTDB.save('OrderValidatorLog', log)


class Broker(ABC):
    '''
    Broker is a base class that manages the connection to a broker system. Extend this class 
    to add a concrete implementation to talk to a particular broker.
    '''

    AVAILABLE_BROKERS = {'IB': 'Interactive Brokers', 'Manual': 'Manual Trader'}
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
        elif account.broker == 'Manual':
            from .manual import ManualBroker
            return ManualBroker(account)
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
    def cancel_order(self, plan: TradePlan, order: RawOrder = None) -> bool:
        '''Cancel a specific order or all pending orders of a trade plan if `order` is None.

        Args:
            plan: A trade plan
            order: A `RawOrder` to be cancelled

        Raises:
            RuntimeError if error occurs during cancellation
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
    def get_portfolio(self) -> pd.DataFrame | None:
        '''Get account portfolio

        Returns:
            A dataframe that contains broker specific portfolio infomation
        '''
        raise NotImplementedError()

    @abstractmethod
    def download_trades(self) -> pd.DataFrame | None:
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
    def download_orders(self) -> pd.DataFrame | None:
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
            A list of dict that contains broker specific trade infomation associated with the order, 
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
