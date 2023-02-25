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
from minitrade.utils.convert import (bytes_to_str, csv_to_df, df_to_str,
                                     iso_to_datetime, obj_to_str)
from minitrade.utils.mtdb import MTDB
from minitrade.utils.providers import mailjet_send_email

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TradePlan:
    id: str
    name: str
    strategy_file: str
    ticker_css: str             # ticker comma separated list
    market_timezone: str
    data_source: str
    backtest_start_date: str
    trade_start_date: str
    trade_time_of_day: str
    broker_account: str | None
    commission_rate: float = 0.001
    initial_cash: float = 10_000
    enabled: bool = False
    create_time: datetime
    update_time: datetime | None
    broker_ticker_map: str | None

    def broker_ticker(self, ticker: str):
        ticker_map = json.loads(self.broker_ticker_map)
        return ticker_map[ticker]

    @staticmethod
    def list_plans() -> list[TradePlan]:
        '''Return the list of available trade plans

        Returns
        -------
        list
            A list of trade plans, or empty list if error or no plan found

        Raises
        ------
        RuntimeError
            If listing trade plans fails
        '''
        return MTDB.get_all('tradeplan', orderby='name', cls=TradePlan)

    @staticmethod
    def get_plan(plan_id: str) -> TradePlan:
        '''Look up a trade plan by ID

        Parameters
        ----------
        plan_id : str
            Trade plan ID

        Returns
        -------
        plan
            Trade plan if found

        Raises
        ------
        RuntimeError
            If looking up trade plans fails
        '''
        return MTDB.get_one('tradeplan', 'id', plan_id, cls=TradePlan)

    def __call_scheduler(self, method: str, path: str, params: dict | None = None) -> Any:
        '''Call the scheduler's REST API

        Parameters
        ----------
        method : str
            The HTTP method, i.e. GET, POST, PUT, DELETE, etc.
        path : str
            The REST API endpoint, e.g. '/jobs'
        params : dict, optional
            Extra parameters to be sent along with the REST API call

        Returns
        -------
        Any
            result json if API returns 200, otherwise None
        '''
        url = f'http://{config.scheduler.host}:{config.scheduler.port}{path}'
        resp = requests.request(method=method, url=url, params=params)
        if resp.status_code == 200:
            return resp.json()

    def enable(self, enable: bool) -> None:
        '''Enable and schedule a plan, or disable and unschedule a plan.

        Parameters
        ----------
        enable : bool
            True to enable, False to disable

        Raises
        ------
        RuntimeError
            If enabling/disabling trade plan fails
        '''
        self.enabled = enable
        self.update_time = datetime.utcnow()
        MTDB.update('tradeplan', 'id', self.id, values={'enabled': self.enabled, 'update_time': self.update_time})

    def jobinfo(self) -> Any:
        '''Get scheduled job info of the trade plan.

        Raises
        ------
        RuntimeError
            If getting job info fails
        '''
        try:
            return self.__call_scheduler('GET', f'/jobs/{self.id}')
        except Exception as e:
            raise RuntimeError(f'Getting trade plan job info {self.id} {self.name} failed') from e

    def save(self) -> None:
        '''Schedule the trade plan and save it to database.

        Overwrite existing ones if plan ID or plan name already exists.

        Raises
        ------
        RuntimeError
            If saving trade plan fails
        '''
        MTDB.save(self, 'tradeplan')
        self.__call_scheduler('PUT', f'/jobs/{self.id}')

    def delete(self) -> None:
        '''Unschedule a trade plan and delete it from database.

        Raises
        ------
        RuntimeError
            If deleting trade plan fails
        '''
        MTDB.delete('tradeplan', 'id', self.id)
        self.__call_scheduler('DELETE', f'/jobs/{self.id}')

    def get_orders(self, runlog_id: str = None) -> list[RawOrder]:
        '''Get intended orders 
        '''
        if runlog_id:
            return MTDB.get_all('raworder', where={'plan_id': self.id, 'runlog_id': runlog_id}, cls=RawOrder)
        else:
            return MTDB.get_all('raworder', 'plan_id', self.id, cls=RawOrder)


def entry_strategy(strategy):
    ''' Decorator to help specify the entry strategy if multiple strategy classes exists in a strategy file'''
    strategy.__entry_strategy__ = True
    return strategy


class StrategyManager:

    __strategy_root = expanduser('~/.minitrade/strategy')

    @staticmethod
    def __path(filename: str) -> str:
        ''' Get the absolute path of a strategy file '''
        return os.path.join(StrategyManager.__strategy_root, filename)

    @staticmethod
    def list_strategies() -> list[str]:
        '''Return the list of available strategy files

        Returns
        -------
        list
            A list of strategy filenames
        '''
        return [f for f in os.listdir(StrategyManager.__strategy_root) if os.path.isfile(StrategyManager.__path(f))]

    @staticmethod
    def save_strategy(filename: str, content: str) -> None:
        ''' Save `content` to a strategy file named `name`. 

        Parameters
        ----------
        name : str
            Strategy name
        content : str
            File content as string
        '''
        module_file = StrategyManager.__path(filename)
        with open(module_file, 'w') as f:
            f.write(content)

    @staticmethod
    def load_strategy(strategy_file: str) -> Strategy:
        '''Load strategy class from `strategy_file`.

        Parameters
        ----------
        strategy_file : str
            The strategy file name relative to the strategy root path

        Returns
        -------
        Strategy
            The loaded strategy class

        Raises
        ------
        RuntimeError
            If a strategy class is not found for any reason.
        '''
        strategy_path = StrategyManager.__path(strategy_file)
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
    def read_strategy(strategy_file: str) -> str:
        ''' Return the content of `strategy_file` '''
        strategy_path = StrategyManager.__path(strategy_file)
        with open(strategy_path, 'r') as f:
            return f.read()

    @staticmethod
    def delete_strategy(strategy_file: str) -> None:
        ''' Delete `strategy_file` '''
        strategy_path = StrategyManager.__path(strategy_file)
        os.remove(strategy_path)


@dataclass(kw_only=True)
class RawOrder:
    id: str
    plan_id: str
    runlog_id: str
    ticker: str
    side: str
    size: int
    signal_date: datetime
    entry_type: str
    broker_order_id: str | None

    def save(self) -> None:
        '''Save an order to database. 

        Overwrite existing ones if order id already exists.

        Parameters
        ----------
        order : RawOrder
            the order
        '''
        try:
            MTDB.save(self, 'raworder', on_conflict='update')
        except Exception as e:
            raise RuntimeError(f'Saving raw order {self} failed') from e

    def tag(self):
        return f'{self.side} {self.ticker} {abs(self.size)}'


@dataclass(kw_only=True)
class BacktestRunLog:
    id: str
    plan_id: str
    plan_name: str
    plan_strategy: str
    plan: str
    data: str | None = None
    strategy_code: str | None
    result: str | None = None
    exception: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    log_time: datetime

    def error(self) -> bool:
        return self.result is None or self.exception is not None or self.stderr is not None


class BacktestRunner:

    def __init__(self, plan: TradePlan):
        self.plan = plan

    def get_price(self) -> pd.DataFrame:
        '''Load quote data from the data source for tickers as specified in the trade plan

        Parameters
        ----------
        plan : TradePlan
            The trade plan which specifies the data source, tickers, and the backtest
            start date from when the OHLCV data should be loaded

        Returns
        -------
        DataFrame
            A dataframe, indexed by date, with two level columns where level 0 is the
            ticker, and level 1 is OHLCV

        Raises
        ------
        DataLoadingError
            If data aren't loaded successfully for any reason
        '''
        try:
            source = QuoteSource.get_source(self.plan.data_source)
            return source.daily_ohlcv(
                tickers=self.plan.ticker_css.split(','),
                start=self.plan.backtest_start_date)
        except Exception as e:
            raise RuntimeError('Getting price data failed') from e

    def get_strategy(self) -> tuple[Strategy, str]:
        '''Load strategy'''
        code = StrategyManager.read_strategy(self.plan.strategy_file)
        strategy = StrategyManager.load_strategy(self.plan.strategy_file)
        return strategy, code

    def run_backtest(self, runlog_id: str = None, dryrun: bool = False, **kwargs) -> pd.Series | None:
        '''Run backtest according to the trade plan and log the result to database.

        Parameters
        ----------
        plan : TradePlan
            The trade plan
        dryrun : bool
            A flag to control if the backtest generates actual trades (default is False)
        runlog_id : str
            A string ID to uniquely identify this backtest run. Leave as None to
            automatically generate one.

        Returns
        -------
        series
            Backtest statistics, orders, trades, etc. or None if backtest is not successful
        '''
        data = code = strategy = result = ex = None
        try:
            runlog_id = runlog_id if runlog_id else MTDB.uniqueid()
            data = self.get_price()
            strategy, code = self.get_strategy()
            bt = Backtest(strategy=strategy, data=data, cash=self.plan.initial_cash,
                          commission=self.plan.commission_rate, trade_start_date=self.plan.trade_start_date, **kwargs)
            result = bt.run()
            if result is not None and not dryrun and self.plan.broker_account is not None:
                self.record_orders(runlog_id, result)
            return result
        except Exception:
            ex = traceback.format_exc()
        finally:
            self.log_backtest_run(runlog_id, data=data, strategy_code=code, result=result, exception=ex)

    def record_orders(self, runlog_id: str, result: pd.Series) -> None:
        '''Record raw orders generated from a backtest run to database'''
        def hash(s: pd.Series):
            # include fields that can uniquely identify an order
            signature = s.loc[['plan_id', 'ticker', 'size', 'signal_date',
                               'entry_type']].to_json(date_format='iso').encode('utf-8')
            return hashlib.md5(signature).hexdigest()

        try:
            # record orders from the backtest result to database
            orders: pd.DataFrame = result['_orders'].reset_index()
            orders.rename(columns={'SignalDate': 'signal_date', 'Ticker': 'ticker',
                                   'Side': 'side', 'Size': 'size', 'EntryType': 'entry_type'}, inplace=True)
            orders['plan_id'] = self.plan.id
            orders['runlog_id'] = runlog_id
            orders['id'] = orders.apply(lambda x: hash(x), axis=1)
            MTDB.save(orders.to_dict('records'), 'raworder', on_conflict='ignore')
        except Exception as e:
            raise RuntimeError(f'Saving orders failed for runlog {runlog_id}') from e

    def execute(self, dryrun: bool = False) -> BacktestRunLog | None:
        '''Run backtest in an isolated process.

        This is the same as `run_backtest` but run it in an isolated process.

        Parameters
        ----------
        plan : TradePlan
            The trade plan

        Returns
        -------
        runlog_id
            The runlog ID that identifies the backtest run if the backtest runs to finish,
            otherwise None
        '''
        try:
            runlog_id = MTDB.uniqueid()
            proc = subprocess.run(
                [sys.executable, '-m', 'minitrade.cli', 'execute', '--execid', runlog_id, self.plan.id] +
                (['--dryrun'] if dryrun else []) + (['--pytest'] if 'pytest' in sys.modules else []),
                capture_output=True,
                cwd=os.getcwd(),
                timeout=600)
            self.log_backtest_run(runlog_id,
                                  stdout=proc.stdout if proc.stdout else None,
                                  stderr=proc.stderr if proc.stderr else None)
            return self.get_backtest_runlog(runlog_id)
        except subprocess.TimeoutExpired as e:
            self.log_backtest_run(runlog_id,
                                  stdout=e.stdout if e.stdout else None,
                                  stderr=e.stderr if e.stderr else None)
            raise RuntimeError(f'Backtest for {self.plan} runlog {runlog_id} took too long to finish, killed') from e
        except Exception as e:
            raise RuntimeError(f'Backtest for {self.plan} runlog {runlog_id} failed') from e

    def log_backtest_run(
            self,
            runlog_id: str,
            data: pd.DataFrame | None = None,
            strategy_code: str | None = None,
            result: pd.Series | None = None,
            exception: str | None = None,
            stdout: bytes | None = None,
            stderr: bytes | None = None) -> None:
        '''Log the input and output of a backtest run to database.

        Parameters
        ----------
        runlog_id: : str
            Backtest run ID
        plan : TradePlan
            The trade plan
        data : DataFrame | None
            Quote data loaded from data source and used in backtest
        strategy_class : Strategy | None
            The strategy class
        result : Series | None
            Backtest result
        exception : Exception | None
            Exception thrown during the backtest run if any
        stdout : bytes
            Stdout output of backtest process
        stderr : bytes
            Stderr output of backtest process

        Raises
        ------
        BacktestRunSavingError
            If the backtest log is not saved successfully for any reason
        '''
        log = self.get_backtest_runlog(runlog_id)
        if log is None:
            log = BacktestRunLog(
                id=runlog_id,
                plan_id=self.plan.id,
                plan_name=self.plan.name,
                plan_strategy=self.plan.strategy_file,
                plan=obj_to_str(self.plan),
                data=df_to_str(data),
                strategy_code=strategy_code,
                result=df_to_str(result),
                exception=exception,
                stdout=bytes_to_str(stdout),
                stderr=bytes_to_str(stderr),
                log_time=datetime.utcnow()
            )
        else:
            log.data = log.data or data
            log.strategy_code = log.strategy_code or strategy_code
            log.result = log.result or result
            log.exception = log.exception or exception
            log.stdout = log.stdout or stdout
            log.stderr = log.stderr or stderr
        try:
            MTDB.save(log, 'backtestrunlog', on_conflict='update')
        except Exception as e:
            raise RuntimeError(f'Saving runlog {log.id} failed') from e
        self.send_summary_email(log)

    def send_summary_email(self, log: BacktestRunLog):
        ''' Send backtest summary email '''
        orders = self.plan.get_orders(log.id)
        ts = log.log_time.strftime("%Y-%m-%d %H:%M:%S")
        status = "failed" if log.error() else "succeeded"
        subject = f'Plan {log.plan_name} {status} @ {ts}' + (f' {len(orders)} new orders' if orders else '')
        data = csv_to_df(log.data, index_col=0, header=[0, 1], parse_dates=True).xs('Close', 1, 1).tail(2).T
        if log.result:
            result = csv_to_df(log.result, index_col=0)
            result = result[~result.index.str.startswith('_')]['0'].to_string()
        else:
            result = None
        message = [
            f'Plan {log.plan_name} @ {ts} {status}',
            '\n'.join([o.tag() for o in orders]) if orders else 'No new orders',
            f'Data\n{data.to_string()}',
            f'Result\n{result}' if result else 'No backtest result',
            f'Exception\n{log.exception}' if log.exception else 'No exception',
            f'Stdout\n{log.stdout}' if log.stdout else 'No stdout',
            f'Stderr\n{log.stderr}' if log.stderr else 'No stderr'
        ]
        message = '\n\n'.join(message)
        mailjet_send_email(subject, message)

    def list_runlogs(self) -> list[BacktestRunLog]:
        '''Return all backtest history for a trade plan.

        Parameters
        ----------
        plan_id : str
            The trade plan ID

        Returns
        -------
        list
            A list of backtest runlogs, or empty list if error or no record found
        '''
        return MTDB.get_all(
            'backtestrunlog', 'plan_id', self.plan.id, orderby=('log_time', False),
            limit=100, cls=BacktestRunLog)

    def get_backtest_runlog(self, runlog_id: str) -> BacktestRunLog:
        '''Return log record for a backtest run.

        Parameters
        ----------
        runlog_id : str
            The backtest run ID

        Returns
        -------
        runlog
            Log of the backtest, or None if not found
        '''
        return MTDB.get_one('backtestrunlog', 'id', runlog_id, cls=BacktestRunLog)

    def print_backtest_result(self, result: pd.Series) -> None:
        '''Break out backtest result into performance statistics, profit and loss, trades, and orders.

        Parameters
        ----------
        result : Series
            The backtest result

        Returns
        -------
        stats
            Performance statistics
        pnl
            Profit and loss per ticker
        trades
            Trades happened
        orders
            Orders placed
        '''
        stats = result.drop(index='_strategy,_equity_curve,_trades,_orders'.split(','))
        pnl = result['_trades'].groupby('Ticker')['PnL'].sum().astype(int).to_frame().T
        return stats, pnl, result['_trades'], result['_orders']


@dataclass(kw_only=True)
class ValidatorRunLog:
    id: str
    trace_id: str
    order: str
    result: str
    exception: str | None = None
    log_time: datetime


class OrderValidator:

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
        order_in_db = MTDB.get_one('raworder', 'id', order.id, cls=RawOrder)
        self.__assert_not_null(order_in_db, 'Order is not in raworder table')
        self.__assert_equal(order, order_in_db, 'Order is not in sync with db')

    def _assert_order_has_correct_plan_id(self, order: RawOrder):
        self.__assert_equal(order.plan_id, self.plan.id, 'Order has incorrect plan ID')

    def _assert_order_has_correct_runlog_id(self, order: RawOrder):
        self.__assert_not_null(MTDB.get_one('backtestrunlog', 'id', order.runlog_id,
                                            cls=BacktestRunLog), 'Order has incorrect runlog ID')

    def _assert_order_has_correct_ticker(self, order: RawOrder):
        self.__assert_is_in(order.ticker, self.plan.ticker_css.split(','), 'Order has incorrect ticker')
        self.__assert_not_null(self.plan.broker_ticker(order.ticker), 'Order has incorrect ticker mapping')

    def _assert_order_has_correct_size(self, order: RawOrder):
        self.__assert((order.size > 0 and order.side == 'Buy') or (
            order.size < 0 and order.side == 'Sell'), 'Order side and size does not agree')

    def _assert_order_size_is_within_limit(self, order: RawOrder):
        self.__assert_less_than(abs(order.size), 10000, 'Order size is too big')

    def _assert_order_in_time_window(self, order: RawOrder):
        plan = TradePlan.get_plan(order.plan_id)
        now = datetime.now(tz=ZoneInfo(plan.market_timezone))
        usmarket = MTDB.get_one('nasdaqtraded', 'symbol', order.ticker) is not None
        self.__assert_equal(usmarket, True, 'Only U.S. market is supported for now')
        if order.entry_type == 'MOO':
            # MOO order submit window is between market close on signal_date and
            # before next market open, considering weekends but not holidays
            market_open, market_close = timedelta(hours=9, minutes=30), timedelta(hours=16)
            self.__assert_less_than(order.signal_date + market_close, now,
                                    'MOO order must be submitted after market close')
            self.__assert_less_than(
                now, order.signal_date + market_open + timedelta(days=1 if order.signal_date.weekday() < 4 else 3),
                'MOO order must be submitted before next market open')
        elif order.entry_type == 'MOC':
            # MOC order submit window is before market close on signal_date
            self.__assert_less_than(now, order.signal_date + market_close,
                                    'MOC order must be submitted before market close')
        else:
            self.__assert(False, f'Unknown order entry type: {order.entry_type}')

    def _assert_order_has_no_broker_order_id(self, order: RawOrder):
        self.__assert_is_null(order.broker_order_id, 'Order already has broker_order_id')

    def _assert_order_is_not_in_iborder_table(self, order: RawOrder):
        iborder = self.broker.find_order(order)
        # Okay if no order exists or order exists but not fully submitted
        self.__assert(iborder is None or iborder['status'] == 'Inactive', 'Order exists in iborder table')

    def _assert_order_is_not_in_ibtrade_table(self, order: RawOrder):
        trades = self.broker.find_trades(order)
        self.__assert_equal(len(trades), 0, 'Order exists in ibtrade table')

    def validate(self, order: RawOrder, trace_id: str = None):
        tests = [
            self._assert_order_is_in_sync_with_db,
            self._assert_order_has_correct_plan_id,
            self._assert_order_has_correct_runlog_id,
            self._assert_order_has_correct_ticker,
            self._assert_order_has_correct_size,
            self._assert_order_size_is_within_limit,
            self._assert_order_in_time_window,
            self._assert_order_has_no_broker_order_id,
            self._assert_order_is_not_in_iborder_table,
            self._assert_order_is_not_in_ibtrade_table,
        ]
        result, ex = {}, None
        try:
            for t in tests:
                result[t.__name__] = 'Pass' if t(order) is None else 'Fail'
        except Exception as e:
            result[t.__name__.removeprefix('_assert_')] = 'Fail'
            ex = traceback.format_exc()
            raise RuntimeError(f'Order {order.id} failed {t.__name__}') from e
        finally:
            self.log_validator_run(trace_id, order, result, ex)

    def log_validator_run(self, trace_id: str, order: RawOrder, result: dict, exception: str) -> None:
        '''Log the input and output of a trader run to database.
        '''
        log = ValidatorRunLog(
            id=MTDB.uniqueid(),
            trace_id=trace_id,
            order=obj_to_str(order),
            result=obj_to_str(result),
            exception=exception,
            log_time=datetime.utcnow()
        )
        MTDB.save(log, 'ordervalidlog', on_conflict='error')


@dataclass(kw_only=True)
class TraderRunLog:
    id: str
    summary: str
    start_time: datetime
    log_time: datetime


class Trader:

    def select_orders_for_trading(self, plan: TradePlan):
        '''Select orders that should be traded, including retries'''
        return MTDB.get_all(
            'raworder', where={'plan_id': plan.id, 'broker_order_id': None},
            orderby='signal_time', cls=RawOrder)

    def place_orders(self, plan: TradePlan, orders: list[RawOrder]) -> None:
        '''Place orders.
        '''
        start_time = datetime.utcnow()
        account = BrokerAccount.get_account(plan)
        broker = Broker.get_broker(account)
        validator = OrderValidator(plan, broker)
        trace_id = MTDB.uniqueid()
        summary = [f'Trace ID {trace_id}', str(plan), f'{len(orders)} orders to be processed']
        try:
            broker.connect()
            summary.append(f'Broker {broker.account.alias} is ready')
            # download orders and trades before submitting new orders
            broker.download_orders()
            broker.download_trades()
            for i, order in enumerate(orders):
                try:
                    order_trace_id = f'{trace_id}-{i}'
                    validator.validate(order, trace_id=order_trace_id)
                    broker_order_id = broker.submit_order(plan, order, trace_id=order_trace_id)
                    order.broker_order_id = broker_order_id
                    order.save()
                except Exception as e:
                    summary.append(f'#{i} {order.ticker} {order.side} {abs(order.size)} ERROR {str(e)}')
                    summary.append('Order processing aborted')
                    break
                else:
                    summary.append(f'#{i} {order.ticker} {order.side} {abs(order.size)} OK ({order.id})')
            # download orders and trades after submitting new orders
            broker.download_orders()
            broker.download_trades()
            self.log_trader_run(trace_id, summary, start_time)
        except ConnectionError as e:
            summary.append(f'Broker {broker.account.alias} is not ready, orders not submitted')
            self.log_trader_run(trace_id, summary, start_time)

    def execute(self):
        for plan in TradePlan.list_plans():
            if plan.enabled:
                try:
                    orders = self.select_orders_for_trading(plan)
                    if orders:
                        self.place_orders(plan, orders)
                except Exception:
                    logger.exception(f'Submitting orders for {plan} failed')

    def log_trader_run(self, trace_id: str, summary: list[str], start_time: datetime) -> None:
        '''Log the input and output of a trader run to database.
        '''
        text = '\n\n'.join(summary)
        log = TraderRunLog(
            id=trace_id,
            summary=text,
            start_time=start_time,
            log_time=datetime.utcnow()
        )
        MTDB.save(log, 'traderlog', on_conflict='error')
        mailjet_send_email(f'Trader @ {start_time} finished', text)
