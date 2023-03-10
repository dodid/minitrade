from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from posixpath import expanduser
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from minitrade.backtest import Backtest, Strategy
from minitrade.broker import Broker, BrokerAccount
from minitrade.datasource import QuoteSource
from minitrade.utils.config import config
from minitrade.utils.mailjet import mailjet_send_email
from minitrade.utils.mtdb import MTDB
from minitrade.utils.telegram import telegram_send_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'TradePlan',
    'entry_strategy',
    'StrategyManager',
    'RawOrder',
    'BacktestLog',
    'BacktestRunner',
    'OrderValidatorLog',
    'TraderLog',
    'Trader',
]


@dataclass(kw_only=True)
class TradePlan:
    '''TradePlan specifies how a strategy should be executed that includes:

    - which strategy to run 
    - the universe of assets as a list of tickers 
    - which data source to get price data from 
    - which timezone are the assets traded 
    - which date should a backtest starts 
    - which date should a trade order be generated from 
    - at what time of day should a backtest run
    - which broker account should orders be submitted through 
    - how much initial cash should be invested
    '''

    id: str
    '''TradePlan ID'''

    name: str
    '''A unique human readable name'''

    strategy_file: str
    '''Strategy file name'''

    ticker_css: str
    '''Asset universe as a list of tickers in form of a comman separated string'''

    market_timezone: str
    '''Timezone of the market where the tickers are traded'''

    data_source: str
    '''Quote source name as in QuoteSource.AVAILABLE_SOURCES'''

    backtest_start_date: str
    '''Since when backtest should start in string format "YYYY-MM-DD".'''

    trade_start_date: str | None
    '''Since when orders should be generated in string format "YYYY-MM-DD".
    Orders generated before the date are ignored.'''

    trade_time_of_day: str
    '''At what time of day should backtest starts in string format "HH:MM:SS".'''

    broker_account: str | None
    '''Alias of the broker account used to submit orders'''

    commission_rate: float = 0.001
    '''Commission ratio related to trade value'''

    initial_cash: float
    '''Cash amount to start with'''

    enabled: bool = False
    '''If the trade plan is enabled for trading'''

    create_time: datetime
    '''Time when the trade plan is created'''

    update_time: datetime | None
    '''Time when the trade plan is updated'''

    broker_ticker_map: str | None
    '''A json string encoding a dict mapping from generic tickers to broker
    specific instrument IDs'''

    def broker_instrument_id(self, ticker: str) -> Any:
        '''Get broker instrument ID corresponding to a generic `ticker`.

        Args:
            ticker: A generic ticker

        Returns:
            A broker specific instrument ID
        '''
        return self.broker_ticker_map[ticker]

    @staticmethod
    def list_plans() -> list[TradePlan]:
        '''Return the list of available trade plans

        Returns:
            A list of zero or more trade plans
        '''
        return MTDB.get_all('TradePlan', orderby='name', cls=TradePlan)

    @staticmethod
    def get_plan(plan_id_or_name: str) -> TradePlan:
        '''Look up a trade plan by plan ID or plan name

        Args:
            plan_id_or_name: Trade plan ID or name

        Returns:
            Trade plan if found or None
        '''
        return MTDB.get_one(
            'TradePlan', 'id', plan_id_or_name, cls=TradePlan) or MTDB.get_one(
            'TradePlan', 'name', plan_id_or_name, cls=TradePlan)

    def __call_scheduler(self, method: str, path: str, params: dict | None = None) -> Any:
        url = f'http://{config.scheduler.host}:{config.scheduler.port}{path}'
        resp = requests.request(method=method, url=url, params=params)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code >= 400:
            raise RuntimeError(f'Scheduler {method} {url} {params} returns {resp.status_code} {resp.text}')

    def enable(self, enable: bool):
        '''Enable or disable a plan for trading.

        Args:
            enable: True to schedule a plan for trading, or False to deschedule.
        '''
        self.enabled = enable
        self.update_time = datetime.utcnow()
        MTDB.update('TradePlan', 'id', self.id, values={'enabled': self.enabled, 'update_time': self.update_time})
        self.__call_scheduler('PUT', f'/jobs/{self.id}')

    def jobinfo(self) -> dict:
        '''Get scheduled job status of the trade plan.

        Returns:
            A dict like {job_id: ..., job_frequency: ..., next_run_time: ...}
        '''
        return self.__call_scheduler('GET', f'/jobs/{self.id}')

    def save(self) -> None:
        '''Schedule the trade plan and save it to database.

        Fail if plan ID or plan name already exists.
        '''
        MTDB.save(self, 'TradePlan')
        self.__call_scheduler('PUT', f'/jobs/{self.id}')

    def delete(self) -> None:
        '''Unschedule a trade plan and delete it from database.
        '''
        MTDB.delete('TradePlan', 'id', self.id)
        self.__call_scheduler('DELETE', f'/jobs/{self.id}')

    def get_orders(self, run_id: str = None) -> list[RawOrder]:
        '''Retrieve all raw orders associated with a trade plan or only those generated during 
        a particular backtest run.

        Args:
            run_id: Backtest run ID. If None, return all orders generated from the trade plan.

        Returns:
            A list of zero or more RawOrders 
        '''
        if run_id:
            return MTDB.get_all('RawOrder', where={'plan_id': self.id, 'run_id': run_id}, cls=RawOrder)
        else:
            return MTDB.get_all('RawOrder', 'plan_id', self.id, cls=RawOrder)

    def list_logs(self) -> list[BacktestLog]:
        '''Return all backtest history for this trade plan.

        Returns:
            A list of zero or more backtest logs
        '''
        return MTDB.get_all(
            'BacktestLog', 'plan_id', self.id, orderby=('log_time', False),
            limit=100, cls=BacktestLog)

    def get_log(self, run_id: str) -> BacktestLog:
        '''Return backtest log for a particular run.

        Args:
            run_id: The backtest run ID

        Returns:
            Backtest log, or None if not found
        '''
        return MTDB.get_one('BacktestLog', 'id', run_id, cls=BacktestLog)


def entry_strategy(strategy):
    ''' Decorator to help specify the entry strategy when multiple strategy classes exists in a strategy file'''
    strategy.__entry_strategy__ = True
    return strategy


class StrategyManager:
    '''StrategyManager provides methods to organizes strategy files.'''

    __strategy_root = expanduser('~/.minitrade/strategy')

    @staticmethod
    def _path(filename: str) -> str:
        ''' Get the absolute path of a strategy file '''
        return os.path.join(StrategyManager.__strategy_root, filename)

    @staticmethod
    def list() -> list[str]:
        '''Return the list of available strategy files

        Returns:
            A list of strategy filenames
        '''
        return [f for f in os.listdir(StrategyManager.__strategy_root) if os.path.isfile(StrategyManager._path(f))]

    @staticmethod
    def save(filename: str, content: str):
        '''Save `content` to a strategy file. 

        Args:
            filename: Strategy file name
            content: File content as string
        '''
        module_file = StrategyManager._path(filename)
        with open(module_file, 'w') as f:
            f.write(content)

    @staticmethod
    def load(strategy_file: str) -> Strategy:
        '''Load strategy class from `strategy_file`.

        Args:
            strategy_file: The strategy file name

        Returns:
            The loaded strategy class

        Raises:
            RuntimeError: If a strategy class is not found for any reason.
        '''
        strategy_path = StrategyManager._path(strategy_file)
        module_name = strategy_file.removesuffix('.py')
        # load the strategy file
        spec = importlib.util.spec_from_file_location('strategy', strategy_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        # look up and return the strategy class
        strategy_found = []
        for attr in dir(module):
            attr_val = getattr(module, attr)
            try:
                if issubclass(attr_val, Strategy) and attr != 'Strategy':
                    strategy_found.append(attr_val)
            except Exception:
                pass
        if len(strategy_found) > 1:
            entry = [s for s in strategy_found if '__entry_strategy__' in s.__dict__]
            if len(entry) == 1:
                return entry[0]
            raise RuntimeError(
                f'Multiple strategy classes found: {strategy_found}. Use @entry_strategy decorator to specify which one should run.')
        elif len(strategy_found) == 1:
            return strategy_found[0]
        else:
            raise RuntimeError(f'No strategy found in {strategy_path}')

    @staticmethod
    def read(strategy_file: str) -> str:
        ''' Return the content of `strategy_file` 

        Args:
            strategy_file: The strategy file name

        Returns:
            The file content as a string
        '''
        strategy_path = StrategyManager._path(strategy_file)
        with open(strategy_path, 'r') as f:
            return f.read()

    @staticmethod
    def delete(strategy_file: str) -> None:
        '''Delete `strategy_file`

        Args:
            strategy_file: The strategy file name
        '''
        strategy_path = StrategyManager._path(strategy_file)
        os.remove(strategy_path)


@dataclass(kw_only=True)
class RawOrder:
    '''RawOrder represents an order generated during the backtest run of a strategy.
    It captures the asset, side, size, and timing of an order among others.
    The order is broker independent until it's submitted via a particular broker
    account, after which a broker order ID can be associated.
    '''

    id: str
    '''Raw order ID'''

    plan_id: str
    '''ID of the trade plan that generates this order'''

    run_id: str
    '''ID of particular backtest run that generates this order'''

    ticker: str
    '''Ticker to buy or sell'''

    side: str
    '''Side to take for the order, "Buy" or "Sell".'''

    size: int
    '''Number of shares to buy or sell, positive for long and negative for short.'''

    signal_time: datetime
    '''Date on which the order signal is generated.'''

    entry_type: str
    '''Type of order entry, 'MOC' for market-on-close order, 'MOO' for market-on-open order'''

    broker_order_id: str | None
    '''Broker assigned order ID after the raw order is submitted.'''

    def save(self):
        '''Save self to database. 

        Overwrite existing ones if order id already exists.
        '''
        MTDB.save(self, 'RawOrder', on_conflict='update')

    @property
    def tag(self):
        '''A short descriptive label'''
        return f'{self.side} {self.ticker} {abs(self.size)}'


@dataclass(kw_only=True)
class BacktestLog:
    '''BacktestLog captures the inputs and outputs of a backtest run for diagnosis.'''

    id: str
    '''Run ID'''

    plan_id: str
    '''Trade plan ID'''

    plan_name: str
    '''Trade plan name'''

    plan_strategy: str
    '''Strategy filename'''

    plan: str
    '''Trade plan seralized in Json'''

    data: pd.DataFrame | None = None
    '''Quote data used in backtest'''

    strategy_code: str | None
    '''Strategy code snapshot'''

    result: pd.Series | None = None
    '''Backtest result'''

    exception: str | None = None
    '''Exception happened during backtesting'''

    stdout: str | None = None
    '''STDOUT capture during backtesting'''

    stderr: str | None = None
    '''STDERR capture during backtesting'''

    log_time: datetime
    '''Log time'''

    @property
    def error(self) -> bool:
        '''A backtest is regarded as failed if any of the following occurs:

        - Backtest generates no result.
        - Exception occurs.
        - STDERR output is not empty.

        Returns:
            True for error, False for success
        '''
        return bool(self.result is None or self.exception or self.stderr)


class BacktestRunner:
    '''BacktestRunner handles the execution of a backtest, which involves preparing data,
    loading strategy, running backtest, recording orders, writing logs, and sending 
    notifications.

    Usually a scheduler, at time as specified in a trade plan, instantiates a `BacktestRunner`,
    and calls its `execute()` method which starts a backtest in an isolated process.
    The process runs the backtest and writes log into database.
    '''

    def __init__(self, plan: TradePlan):
        self.plan = plan
        self.run_id = None
        self.data = None
        self.strategy = None
        self.code = None
        self.strategy = None
        self.result = None

    def run_backtest(self, run_id: str = None, dryrun: bool = False, **kwargs: dict[str, Any]) -> pd.Series | None:
        '''Run backtest according to the trade plan and log the result to database.

        Args:
            run_id: A string ID to uniquely identify this backtest run. Leave as None to
                automatically generate one.
            dryrun: True to backtest in dryrun mode, in which case no orders are generated.
            **kwargs: Keyword parameters that are passed to Backtest.run().

        Returns:
            Backtest result as pd.Series or None if backtest is not successful
        '''

        exception = None
        try:
            self.run_id = run_id if run_id else MTDB.uniqueid()
            source = QuoteSource.get_source(self.plan.data_source)
            self.data = source.daily_bar(tickers=self.plan.ticker_css, start=self.plan.backtest_start_date)
            self.code = StrategyManager.read(self.plan.strategy_file)
            self.strategy = StrategyManager.load(self.plan.strategy_file)
            bt = Backtest(strategy=self.strategy, data=self.data, cash=self.plan.initial_cash,
                          commission=self.plan.commission_rate, trade_start_date=self.plan.trade_start_date, **kwargs)
            self.result = bt.run()
            if self.result is not None and not dryrun and self.plan.broker_account is not None:
                self._record_orders()
            return self.result
        except Exception:
            exception = traceback.format_exc()
        finally:
            self._log_backtest_run(exception=exception)

    def _record_orders(self) -> None:
        '''Record raw orders from a backtest run to database.

        Backtest is expected to be repeatable, i.e., the orders generated by two backtest runs
        are exactly the same for the period when the two runs overlap. Then the backtest run on
        a later day will possibly generates some new orders based on new data not seen by the 
        earlier run. Such new orders will be recorded in associated with the later run.
        '''

        def hash(s: pd.Series):
            # include fields that can uniquely identify an order
            signature = s.loc[['plan_id', 'ticker', 'size', 'signal_time',
                               'entry_type']].to_json(date_format='iso').encode('utf-8')
            return hashlib.md5(signature).hexdigest()

        # record orders from the backtest result to database
        orders: pd.DataFrame = self.result['_orders'].reset_index()
        orders.rename(columns={'SignalTime': 'signal_time', 'Ticker': 'ticker',
                               'Side': 'side', 'Size': 'size', 'EntryType': 'entry_type'}, inplace=True)
        orders['plan_id'] = self.plan.id
        orders['run_id'] = self.run_id
        orders['id'] = orders.apply(lambda x: hash(x), axis=1)
        MTDB.save(orders.to_dict('records'), 'RawOrder', on_conflict='ignore')

    def execute(self, dryrun: bool = False) -> BacktestLog:
        '''Run backtest in an isolated process.

        Args:
            dryrun: True to run backtest in dryrun mode, in which case no orders are generated.

        Returns:
            `BacktestLog` of the run
        '''
        try:
            self.run_id = MTDB.uniqueid()
            proc = subprocess.run(
                [sys.executable, '-m', 'minitrade.cli', 'backtest', '--run_id', self.run_id, self.plan.id] +
                (['--dryrun'] if dryrun else []) + (['--pytest'] if 'pytest' in sys.modules else []),
                capture_output=True,
                cwd=os.getcwd(),
                timeout=600)
            self._log_backtest_run(stdout=proc.stdout if proc.stdout else None,
                                   stderr=proc.stderr if proc.stderr else None)
            return self.plan.get_log(self.run_id)
        except Exception as e:
            ex = traceback.format_exc()
            self._log_backtest_run(exception=ex,
                                   stdout=proc.stdout if proc.stdout else None,
                                   stderr=proc.stderr if proc.stderr else None)
            raise RuntimeError(f'Backtest for {self.plan} run {self.run_id} failed') from e
        finally:
            self._send_summary_email()

    def _log_backtest_run(self,
                          exception: str = None,
                          stdout: bytes | None = None,
                          stderr: bytes | None = None) -> None:
        '''Log the input and output of a backtest run to database.'''
        log = self.plan.get_log(self.run_id)
        if log is None:
            log = BacktestLog(
                id=self.run_id,
                plan_id=self.plan.id,
                plan_name=self.plan.name,
                plan_strategy=self.plan.strategy_file,
                plan=self.plan,
                data=self.data,
                strategy_code=self.code,
                result=self.result,
                exception=exception,
                stdout=stdout,
                stderr=stderr,
                log_time=datetime.utcnow()
            )
            MTDB.save(log, 'BacktestLog', on_conflict='error')
        else:
            MTDB.update('BacktestLog', 'id', log.id, values={
                'exception': '\n\n'.join([x for x in (log.exception, exception) if x]) or None,
                'stdout': stdout,
                'stderr': stderr,
            })

    def _send_summary_email(self):
        ''' Send backtest summary email '''
        log = self.plan.get_log(self.run_id)
        if log:
            orders = self.plan.get_orders(log.id)
            ts = log.log_time
            status = "failed" if log.error else "succeeded"
            subject = f'Plan {log.plan_name} {status} @ {ts}' + (f' {len(orders)} new orders' if orders else '')
            data = log.data.xs('Close', 1, 1).tail(2).T
            if log.result is not None:
                result = log.result
                result = result[~result.index.str.startswith('_')]['0'].to_string()
            else:
                result = None
            message = [
                f'Plan {log.plan_name} @ {ts} {status}',
                '\n'.join([o.tag for o in orders]) if orders else 'No new orders',
                f'Data\n{data.to_string()}',
                f'Result\n{result}' if result else 'No backtest result',
                f'Exception\n{log.exception}' if log.exception else 'No exception',
                f'Stdout\n{log.stdout}' if log.stdout else 'No stdout',
                f'Stderr\n{log.stderr}' if log.stderr else 'No stderr'
            ]
            message = '\n\n'.join(message)
            telegram_send_message(message)
            mailjet_send_email(subject, message)


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

    def __init__(self, plan: TradePlan, broker: Broker):
        self.plan = plan
        self.broker = broker

    def __assert(self, condition, message):
        if condition != True:
            raise ValueError(message)

    def __assert_is_null(self, value, message):
        return self.__assert(value is None, f'{message}: {value=}')

    def __assert_not_null(self, value, message):
        return self.__assert(value is not None, f'{message}: {value=}')

    def __assert_equal(self, value, expected_value, message):
        return self.__assert(value == expected_value, f'{message}: {value=} != {expected_value=}')

    def __assert_is_in(self, value, collection, message):
        return self.__assert(value in collection, f'{message}: {value=} not in {collection=}')

    def __assert_less_than(self, value, limit, message):
        return self.__assert(value < limit, f'{message}: {value=} >= {limit=}')

    def _assert_order_is_in_sync_with_db(self, order: RawOrder):
        order_in_db = MTDB.get_one('RawOrder', 'id', order.id, cls=RawOrder)
        self.__assert_not_null(order_in_db, 'Order is not in RawOrder table')
        self.__assert_equal(order, order_in_db, 'Order is not in sync with db')

    def _assert_order_has_correct_plan_id(self, order: RawOrder):
        self.__assert_equal(order.plan_id, self.plan.id, 'Order has incorrect plan ID')

    def _assert_order_has_correct_run_id(self, order: RawOrder):
        self.__assert_not_null(MTDB.get_one('BacktestLog', 'id', order.run_id,
                                            cls=BacktestLog), 'Order has incorrect run ID')

    def _assert_order_has_correct_ticker(self, order: RawOrder):
        self.__assert_is_in(order.ticker, self.plan.ticker_css.split(','), 'Order has incorrect ticker')
        self.__assert_not_null(self.plan.broker_instrument_id(order.ticker), 'Order has incorrect ticker mapping')

    def _assert_order_has_correct_size(self, order: RawOrder):
        self.__assert((order.size > 0 and order.side == 'Buy') or (
            order.size < 0 and order.side == 'Sell'), 'Order side and size does not agree')

    def _assert_order_size_is_within_limit(self, order: RawOrder):
        self.__assert_less_than(abs(order.size), 10000, 'Order size is too big')

    def _assert_order_in_time_window(self, order: RawOrder):
        plan = TradePlan.get_plan(order.plan_id)
        now = datetime.now(tz=ZoneInfo(plan.market_timezone))
        usmarket = MTDB.get_one('NasdaqTraded', 'symbol', order.ticker) is not None
        self.__assert_equal(usmarket, True, 'Only U.S. market is supported for now')
        if order.entry_type == 'MOO':
            # MOO order submit window is between market close on signal_time and
            # before next market open, considering weekends but not holidays
            market_open, market_close = timedelta(hours=9, minutes=30), timedelta(hours=16)
            self.__assert_less_than(order.signal_time + market_close, now,
                                    'MOO order must be submitted after market close')
            self.__assert_less_than(
                now, order.signal_time + market_open + timedelta(days=1 if order.signal_time.weekday() < 4 else 3),
                'MOO order must be submitted before next market open')
        elif order.entry_type == 'MOC':
            # MOC order submit window is before market close on signal_time
            self.__assert_less_than(now, order.signal_time + market_close,
                                    'MOC order must be submitted before market close')
        else:
            self.__assert(False, f'Unknown order entry type: {order.entry_type}')

    def _assert_order_has_no_broker_order_id(self, order: RawOrder):
        self.__assert_is_null(order.broker_order_id, 'Order already has broker_order_id')

    def _assert_order_not_in_open_orders(self, order: RawOrder):
        broker_order = self.broker.find_order(order)
        # Okay if no order exists or order exists but not fully submitted
        self.__assert(broker_order is None or broker_order['status'] == 'Inactive', 'Order exists in IbOrder table')

    def _assert_order_not_in_finished_trades(self, order: RawOrder):
        broker_trades = self.broker.find_trades(order)
        self.__assert_equal(len(broker_trades), 0, 'Order exists in IbTrade table')

    def validate(self, order: RawOrder):
        '''Run a bunch of checks to ensure the raw order is valid.

        For example, a valid order should:

        - have valid values for all its attributes
        - not be a duplicate of what has been submitted before
        - be timely
        '''
        tests = [
            self._assert_order_is_in_sync_with_db,
            self._assert_order_has_correct_plan_id,
            self._assert_order_has_correct_run_id,
            self._assert_order_has_correct_ticker,
            self._assert_order_has_correct_size,
            self._assert_order_size_is_within_limit,
            self._assert_order_in_time_window,
            self._assert_order_has_no_broker_order_id,
            self._assert_order_not_in_open_orders,
            self._assert_order_not_in_finished_trades,
        ]
        result, ex = {}, None
        try:
            for test in tests:
                name = test.__name__.removeprefix('_assert_')
                test(order)
                result[name] = 'Pass'
        except Exception as e:
            result[name] = 'Fail'
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
        MTDB.save(log, 'OrderValidatorLog')


@dataclass(kw_only=True)
class TraderLog:
    id: str
    summary: str
    start_time: datetime
    log_time: datetime


class Trader:
    '''Trader handles order submission.

    A scheduler invokes Trader periodically, which checks for new orders generated from different 
    backtest runs, validates, and submits them to brokers.
    '''

    def submit_orders(self, plan: TradePlan):
        '''Submit orders for a particular trade plan. 

        Args:
            plan: Trade plan
        '''
        orders = MTDB.get_all(
            'RawOrder', where={'plan_id': plan.id, 'broker_order_id': None},
            orderby='signal_time', cls=RawOrder)

        if not orders:
            return

        start_time = datetime.utcnow()
        account = BrokerAccount.get_account(plan)
        broker = Broker.get_broker(account)

        if not broker.is_ready():
            telegram_send_message(
                f'Plan {plan.name}',
                f'{len(orders)} order to be submitted:',
                *['- ' + x.tag for x in orders],
                'Use "/ib login" to login.'
            )
            return

        validator = OrderValidator(plan, broker)
        run_id = MTDB.uniqueid()
        summary = [f'Trader run ID {run_id}', f'Plan {plan.name}', f'{len(orders)} order to be processed']

        try:
            # download the latest orders and trades from broker before submitting new orders
            broker.download_orders()
            broker.download_trades()
            for i, order in enumerate(orders):
                try:
                    validator.validate(order)
                    broker_order_id = broker.submit_order(plan, order)
                    order.broker_order_id = broker_order_id
                    order.save()
                except Exception as e:
                    summary.append(f'#{i} {order.ticker} {order.side} {abs(order.size)} ERROR')
                    summary.append(f'  - {str(e)}')
                    summary.append('Order processing aborted')
                    break
                else:
                    summary.append(f'#{i} {order.ticker} {order.side} {abs(order.size)} OK')
            # download orders and trades after submitting new orders
            broker.download_orders()
            broker.download_trades()
        except Exception:
            ex = traceback.format_exc()
            summary.append(ex)
        finally:
            self._log_trader_run(run_id, summary, start_time)

    def execute(self):
        '''Iterate all trade plans and submit orders if there are any new ones.

        Disabled trade plans are not processed.
        '''
        for plan in TradePlan.list_plans():
            if plan.enabled:
                self.submit_orders(plan)

    def _log_trader_run(self, run_id: str, summary: list[str], start_time: datetime) -> None:
        '''Log the input and output of a trader run to database.'''
        text = '\n\n'.join(summary)
        log = TraderLog(
            id=run_id,
            summary=text,
            start_time=start_time,
            log_time=datetime.utcnow()
        )
        MTDB.save(log, 'TraderLog', on_conflict='error')
        telegram_send_message(text)
        mailjet_send_email(f'Trader @ {start_time} finished', text)
