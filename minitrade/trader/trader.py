from __future__ import annotations

import hashlib
import html
import importlib
import json
import logging
import os
import subprocess
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from posixpath import expanduser
from typing import Any
from zoneinfo import ZoneInfo

import dill
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
from tabulate import tabulate

from minitrade.backtest import Backtest, Strategy
from minitrade.backtest.utils import calculate_positions
from minitrade.broker import Broker, BrokerAccount
from minitrade.datasource import QuoteSource
from minitrade.utils.config import config
from minitrade.utils.mailjet import mailjet_send_email
from minitrade.utils.mtdb import MTDB
from minitrade.utils.telegram import send_telegram_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'TradePlan',
    'entry_strategy',
    'StrategyManager',
    'RawOrder',
    'BacktestLog',
    'BacktestRunner',
    'TraderLog',
    'Trader',
    'TaskManager',
    'TaskPlan',
    'TaskLog',
    'TaskRunner',
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
    - when order should be executed, i.e. on market open or on market close
    - which broker account should orders be submitted through 
    - how much initial cash should be invested
    '''

    __storage_root = expanduser('~/.minitrade/storage')

    id: str
    '''TradePlan ID'''

    name: str
    '''A unique human readable name'''

    strategy_file: str
    '''Strategy file name'''

    ticker_css: str
    '''Asset universe as a list of tickers in form of a comman separated string'''

    market_calendar: str
    '''Market calendar name as in pandas_market_calendars.get_calendar_names()'''

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

    entry_type: str
    '''Type of order entry, 'TOO' for trade-on-open order, 'TOC' for trade-on-close order'''

    broker_account: str | None
    '''Alias of the broker account used to submit orders'''

    commission_rate: float = 0.001
    '''Commission ratio related to trade value'''

    initial_cash: float
    '''Cash amount to start with'''

    initial_holding: dict | None = None
    '''Asset positions to start with'''

    strict: bool = True
    '''If True, plan is traded in strict mode'''

    enabled: bool = False
    '''If the trade plan is enabled for trading'''

    create_time: datetime
    '''Time when the trade plan is created'''

    update_time: datetime | None
    '''Time when the trade plan is updated'''

    broker_ticker_map: dict | None = None
    '''A json string encoding a dict mapping from generic tickers to broker
    specific instrument IDs'''

    storage: dict | None = None
    '''A dict to store custom data from backtest runs'''

    def broker_instrument_id(self, ticker: str) -> Any:
        '''Get broker instrument ID corresponding to a generic `ticker`.

        Args:
            ticker: A generic ticker

        Returns:
            A broker specific instrument ID
        '''
        return self.broker_ticker_map[ticker]

    @staticmethod
    def list() -> list[TradePlan]:
        '''Return the list of available trade plans

        Returns:
            A list of zero or more trade plans
        '''
        plans = MTDB.get_all('TradePlan', orderby='name', cls=TradePlan)
        for plan in plans:
            plan.load_storage()
        return plans

    @staticmethod
    def get_plan(plan_id_or_name: str) -> TradePlan:
        '''Look up a trade plan by plan ID or plan name

        Args:
            plan_id_or_name: Trade plan ID or name

        Returns:
            Trade plan if found or None
        '''
        plan: TradePlan = (MTDB.get_one('TradePlan', 'id', plan_id_or_name, cls=TradePlan)
                           or MTDB.get_one('TradePlan', 'name', plan_id_or_name, cls=TradePlan))
        if plan:
            plan.load_storage()
        return plan

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
        self.__call_scheduler('PUT', f'/strategy/{self.id}')

    def jobinfo(self) -> dict:
        '''Get scheduled job status of the trade plan.

        Returns:
            A dict like {job_id: ..., job_frequency: ..., next_run_time: ...}
        '''
        return self.__call_scheduler('GET', f'/strategy/{self.id}')

    def save(self) -> None:
        '''Schedule the trade plan and save it to database.

        Fail if plan ID or plan name already exists.
        '''
        MTDB.save('TradePlan', self, blacklist=['storage'], on_conflict='update')
        self.__call_scheduler('PUT', f'/strategy/{self.id}')

    def delete(self) -> None:
        '''Unschedule a trade plan and delete it from database.
        '''
        MTDB.delete('TradePlan', 'id', self.id)
        try:
            os.remove(os.path.join(TradePlan.__storage_root, f'{self.id}.pkl'))
        except FileNotFoundError:
            pass
        self.__call_scheduler('DELETE', f'/strategy/{self.id}')

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

    def cancel_pending_orders(self) -> None:
        '''Cancel all unsubmitted orders associated with a trade plan.
        '''
        orders = self.get_orders()
        for order in orders:
            if not order.broker_order_id:
                order.cancelled = True
                order.save()

    def list_logs(self, limit=30) -> list[BacktestLog]:
        '''Return all backtest history for this trade plan.

        Returns:
            A list of zero or more backtest logs
        '''
        return MTDB.get_all(
            'BacktestLog', 'plan_id', self.id, orderby=('log_time', False),
            limit=limit, cls=BacktestLog)

    def get_log(self, run_id: str) -> BacktestLog:
        '''Return backtest log for a particular run.

        Args:
            run_id: The backtest run ID

        Returns:
            Backtest log, or None if not found
        '''
        return MTDB.get_one('BacktestLog', 'id', run_id, cls=BacktestLog)

    def load_storage(self):
        '''Load storage from file.'''
        pklfile = os.path.join(TradePlan.__storage_root, f'{self.id}.pkl')
        if os.path.exists(pklfile):
            with open(pklfile, 'rb') as storage_file:
                self.storage = dill.load(storage_file)
        else:
            self.storage = {}

    def save_storage(self):
        '''Save storage to file.'''
        pklfile = os.path.join(TradePlan.__storage_root, f'{self.id}.pkl')
        with open(pklfile, 'wb') as storage_file:
            dill.dump(self.storage or {}, storage_file)


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

    cancelled: bool = False
    '''True if the order is cancelled before submitting.'''

    broker_order_id: str | None
    '''Broker assigned order ID after the raw order is submitted.'''

    def save(self):
        '''Save self to database. 

        Overwrite existing ones if order id already exists.
        '''
        MTDB.save('RawOrder', self, on_conflict='update')

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

    params: str | None
    '''Backtest parameters'''

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

    def save(self):
        MTDB.save('BacktestLog', self, on_conflict='error')


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
        self.params = None

    def _check_data(self, data: pd.DataFrame):
        '''Check if price data is valid.'''
        # Check if data is invariant with previous backtest data.
        if self.plan.strict:
            logs = self.plan.list_logs()
            log = next((l for l in logs if l.data is not None and not l.error), None)
            if log:
                # exclude the last data point which may change intraday
                # only check for price change as volume data from Yahoo do change sometimes
                prefix_len = len(log.data) - 1
                for col in ['Open', 'High', 'Low', 'Close']:
                    if not np.allclose(
                            log.data.xs(col, 1, 1).iloc[: prefix_len],
                            data.xs(col, 1, 1).iloc[: prefix_len],
                            rtol=1e-5):
                        raise RuntimeError(
                            'Data change detected. If this is due to dividend or stock split, please start a new trade plan.')
        # Check if most recent data are actually updated, otherwise issue a warning
        if len(data) > 1:
            same = np.isclose(data.iloc[-1], data.iloc[-2], rtol=1e-5)
            if same.any():
                print('Warning: Data may not be updated. Please verify.')
                print(data.iloc[-2:])

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

        def _record_backtest_params(**params):
            '''Record backtest run to database.'''
            bt = Backtest(**params)
            params.pop('strategy')
            params.pop('data')
            self.params = deepcopy(params)
            return bt

        def _get_trade_start_date():
            '''Trade start date is today if today is a valid trading day and it's after market open time. 
            Otherwise, it's the last valid trading day.
            '''
            calendar = mcal.get_calendar(self.plan.market_calendar)
            today = datetime.now(ZoneInfo(self.plan.market_timezone))
            valid_days = calendar.valid_days(start_date=(today - timedelta(days=30)).strftime('%Y-%m-%d'),
                                             end_date=today.strftime('%Y-%m-%d'), tz=self.plan.market_timezone)
            if today.replace(hour=0, minute=0, second=0, microsecond=0) in valid_days:
                if today.time() > calendar.open_time:
                    trade_start_date = today.strftime('%Y-%m-%d')
                else:
                    trade_start_date = valid_days[-2].strftime('%Y-%m-%d')
            else:
                trade_start_date = valid_days[-1].strftime('%Y-%m-%d')
            return trade_start_date

        exception = None
        try:
            self.run_id = run_id if run_id else MTDB.uniqueid()
            source = QuoteSource.get_source(self.plan.data_source)
            self.data = source.daily_bar(tickers=self.plan.ticker_css, start=self.plan.backtest_start_date)
            self.code = StrategyManager.read(self.plan.strategy_file)
            self.strategy = StrategyManager.load(self.plan.strategy_file)
            self._check_data(self.data)
            if self.plan.strict:
                bt = _record_backtest_params(strategy=self.strategy,
                                             data=self.data,
                                             cash=self.plan.initial_cash,
                                             holding=self.plan.initial_holding,
                                             commission=self.plan.commission_rate,
                                             trade_on_close=self.plan.entry_type == 'TOC',
                                             trade_start_date=self.plan.trade_start_date,
                                             **kwargs)
                self.result = bt.run()
            else:
                account = BrokerAccount.get_account(self.plan)
                broker = Broker.get_broker(account)
                # get the latest portfolio info from broker
                broker.connect()
                broker.download_orders()
                broker.download_trades()
                orders = self.plan.get_orders()
                trades = broker.format_trades(orders)
                current_holding, current_cash = calculate_positions(self.plan, trades)
                # get backtest start date
                trade_start = _get_trade_start_date()
                # cancel all raw orders not yet submitted
                self.plan.cancel_pending_orders()
                # cancel all orders submitted but not yet filled
                broker.cancel_order(self.plan)
                # run backtest
                bt = _record_backtest_params(strategy=self.strategy,
                                             data=self.data,
                                             cash=current_cash,
                                             holding=current_holding,
                                             commission=self.plan.commission_rate,
                                             trade_on_close=self.plan.entry_type == 'TOC',
                                             trade_start_date=trade_start,
                                             storage=self.plan.storage,
                                             **kwargs)
                self.result = bt.run()
            if self.result is not None and not dryrun:
                self._record_orders(ignore_run_id=self.plan.strict)
                self.plan.save_storage()
            return self.result
        except Exception:
            exception = traceback.format_exc()
        finally:
            self._log_backtest_run(exception=exception)

    def _record_orders(self, ignore_run_id) -> None:
        '''Record raw orders from a backtest run to database.

        Backtest is expected to be repeatable in strict mode, i.e., the orders generated by two backtest runs
        are exactly the same for the period when the two runs overlap. Then the backtest run on
        a later day will possibly generates some new orders based on new data not seen by the 
        earlier run. Such new orders will be recorded in associated with the later run.
        '''

        def hash(s: pd.Series):
            # include fields that can uniquely identify an order
            if ignore_run_id:
                df = s.loc[['plan_id', 'ticker', 'size', 'signal_time']]
            else:
                df = s.loc[['run_id', 'plan_id', 'ticker', 'size', 'signal_time']]
            signature = df.to_json(date_format='iso').encode('utf-8')
            return hashlib.md5(signature).hexdigest()

        # record orders from the backtest result to database
        orders: pd.DataFrame = self.result['_orders'].reset_index()
        orders.rename(columns={'SignalTime': 'signal_time', 'Ticker': 'ticker',
                               'Side': 'side', 'Size': 'size'}, inplace=True)
        orders['plan_id'] = self.plan.id
        orders['run_id'] = self.run_id
        orders['id'] = orders.apply(lambda x: hash(x), axis=1)
        orders['cancelled'] = False
        MTDB.save('RawOrder', orders.to_dict('records'), on_conflict='ignore')

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
            self._send_backtest_notification()

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
                params=self.params,
                result=self.result,
                exception=exception,
                stdout=stdout,
                stderr=stderr,
                log_time=datetime.utcnow()
            )
            log.save()
        else:
            MTDB.update('BacktestLog', 'id', log.id, values={
                'exception': '\n\n'.join([x for x in (log.exception, exception) if x]) or None,
                'stdout': stdout,
                'stderr': stderr,
            })

    def _send_backtest_notification(self):
        ''' Send backtest result via telegram and/or email '''
        log = self.plan.get_log(self.run_id)
        if log:
            orders = self.plan.get_orders(log.id)
            plan_name = log.plan_name
            plan_status = 'Failed' if log.error else 'Succeeded'
            plan_time = log.log_time.replace(tzinfo=ZoneInfo('UTC')).astimezone(
                ZoneInfo(self.plan.market_timezone)).strftime('%Y-%m-%d %H:%M:%S')
            plan_subject = f'## {plan_name} ##  {plan_status} @ {plan_time}'
            plan_positions = plan_data = plan_result = plan_exception = plan_stderr = plan_stdout = None

            if log.data is not None:
                data = log.data.xs('Close', 1, 1).tail(2).T
                data.columns = pd.to_datetime(data.columns).strftime('%Y-%m-%d')
                plan_data = tabulate(data, stralign="right", headers='keys')

            if log.result is not None:
                if '_positions' in log.result.index:
                    positions = log.result.loc['_positions'][0]
                    positions = json.loads(positions.replace("'", '"'))
                    plan_positions = tabulate(positions.items(), headers=['Ticker', 'Share'])
                result = log.result
                result = result[~result.index.str.startswith('_')].T.apply(pd.to_numeric, errors='ignore')
                result = result.round(2).T
                for item in ['Start', 'End']:
                    result.loc[item] = result.loc[item][0][:10]
                for item in [
                    'Duration', 'Max. Drawdown Duration', 'Avg. Drawdown Duration', 'Max. Trade Duration',
                        'Avg. Trade Duration']:
                    result.loc[item] = str(result.loc['Duration'][0])[:-10]
                plan_result = tabulate(result, stralign="right")

            plan_exception = log.exception
            plan_stderr = log.stderr
            plan_stdout = log.stdout

            summary = {
                'Status': [plan_status],
                'Exception': [plan_exception is not None],
                'Stderr': [plan_stderr is not None],
                'Stdout': [plan_stdout is not None],
            }

            message = [f'<b>{plan_subject}</b>']
            message.append(f'<pre>{html.escape(tabulate(summary, headers="keys"))}</pre>')

            if orders:
                plan_orders = tabulate([[o.side, o.ticker, abs(o.size)] for o in orders])
                message.append(f'<b>Orders</b>\n<pre>{html.escape(plan_orders)}</pre>')
            else:
                message.append(f'<b>Orders</b>\n<pre>No order generated</pre>')

            if plan_positions:
                message.append(f'<b>Positions</b>\n<pre>{html.escape(plan_positions)}</pre>')

            if plan_data:
                message.append(f'<b>Data</b>\n<pre>{html.escape(plan_data)}</pre>')

            if plan_result:
                message.append(f'<b>Result</b>\n<pre>{html.escape(plan_result)}</pre>')

            if plan_exception:
                message.append(f'<b>Exception</b>\n<pre>{html.escape(plan_exception)}</pre>')

            if plan_stderr:
                message.append(f'<b>Stderr</b>\n<pre>{html.escape(plan_stderr)}</pre>')

            if plan_stdout:
                message.append(f'<b>Stdout</b>\n<pre>{html.escape(plan_stdout)}</pre>')

            message = '\n\n'.join(message)
            # send silent message if the plan runs multiple times a day and there is no error and no order generated
            silent = not log.error and not orders and len(self.plan.trade_time_of_day.split(',')) > 1
            send_telegram_message(html=message, silent=silent)
            if not silent:
                mailjet_send_email(plan_subject, message)


@dataclass(kw_only=True)
class TraderLog:
    id: str
    summary: str
    start_time: datetime
    log_time: datetime

    def save(self):
        MTDB.save('TraderLog', self, on_conflict='error')


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
            'RawOrder', where={'plan_id': plan.id, 'broker_order_id': None, 'cancelled': False},
            orderby='signal_time', cls=RawOrder)

        if not orders:
            return

        start_time = datetime.utcnow()
        account = BrokerAccount.get_account(plan)
        broker = Broker.get_broker(account)

        if not broker.is_ready():
            send_telegram_message(
                f'Plan {plan.name}',
                f'{len(orders)} order to be submitted:',
                *['- ' + x.tag for x in orders],
                'Use "/ib login" to login.'
            )
            return

        run_id = MTDB.uniqueid()
        summary = [f'<b>Trader ## {plan.name} ##</b><pre> </pre>']

        try:
            # download the latest orders and trades from broker before submitting new orders
            broker.download_orders()
            broker.download_trades()
            # submit orders
            status = [[o.ticker, o.side, abs(o.size), 'ABORT'] for o in orders]
            ex = None
            for i, order in enumerate(orders):
                try:
                    order.broker_order_id = broker.submit_order(plan, order)
                    order.save()
                except Exception:
                    status[i][3] = 'ERROR'
                    ex = traceback.format_exc()
                    break
                else:
                    status[i][3] = 'OK'
            summary.append(
                f'<b>Submit orders</b>\n<pre>{html.escape(tabulate(status))}</pre>')
            if ex:
                summary.append(f'<b>Exception</b>\n<pre>{html.escape(ex)}</pre>')
            # download orders and trades after submitting new orders
            broker.download_orders()
            broker.download_trades()
        except Exception:
            ex = traceback.format_exc()
            summary.append(f'<b>Exception</b>\n<pre>{html.escape(ex)}</pre>')
        finally:
            self._log_trader_run(run_id, summary, start_time)

    def execute(self):
        '''Iterate all trade plans and submit orders if there are any new ones.

        Disabled trade plans are not processed.
        '''
        for plan in TradePlan.list():
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
        log.save()
        send_telegram_message(html=text)
        mailjet_send_email(
            f'Trader @ {start_time.replace(tzinfo=ZoneInfo("UTC")).astimezone().strftime("%Y-%m-%d %H:%M:%S")}', text)


@dataclass(kw_only=True)
class TaskLog:

    id: str
    '''Task run ID'''

    plan_id: str
    '''Task plan ID'''

    plan_name: str
    '''Task plan name'''

    task_code: str
    '''Task code snapshot'''

    return_value: int | None = None
    '''Return value of the task'''

    stdout: str | None = None
    '''STDOUT capture during task execution'''

    stderr: str | None = None
    '''STDERR capture during task execution'''

    log_time: datetime
    '''Log time'''

    @property
    def error(self) -> bool:
        '''A task is regarded as failed if return value is not 0

        Returns:
            True for error, False for success
        '''
        return self.return_value != 0

    def save(self):
        MTDB.save('TaskLog', self, on_conflict='error')


@dataclass(kw_only=True)
class TaskPlan:

    id: str
    '''Task plan ID'''

    name: str
    '''A unique human readable name'''

    task_file: str
    '''Task file name'''

    timezone: str
    '''Timezone of the task plan'''

    schedule: str | None
    '''Crontabs of when the task plan should run'''

    notification: str | None
    '''Notification settings in JSON format'''

    enabled: bool = False
    '''If the task plan is enabled'''

    create_time: datetime
    '''Time when the task plan is created'''

    update_time: datetime | None
    '''Time when the task plan is updated'''

    @staticmethod
    def list() -> list[TaskPlan]:
        '''Return the list of available task plan

        Returns:
            A list of zero or more task plans
        '''
        return MTDB.get_all('TaskPlan', orderby='name', cls=TaskPlan)

    @staticmethod
    def get_plan(plan_id_or_name: str) -> TaskPlan:
        '''Look up a task by plan ID or plan name

        Args:
            plan_id_or_name: Task plan ID or name

        Returns:
            Task plan if found or None
        '''
        plan: TaskPlan = (MTDB.get_one('TaskPlan', 'id', plan_id_or_name, cls=TaskPlan)
                          or MTDB.get_one('TaskPlan', 'name', plan_id_or_name, cls=TaskPlan))
        return plan

    def __call_scheduler(self, method: str, path: str, params: dict | None = None) -> Any:
        url = f'http://{config.scheduler.host}:{config.scheduler.port}{path}'
        resp = requests.request(method=method, url=url, params=params)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code >= 400:
            raise RuntimeError(f'Scheduler {method} {url} {params} returns {resp.status_code} {resp.text}')

    def enable(self, enable: bool):
        '''Enable or disable a plan.

        Args:
            enable: True to schedule a plan for running, or False to deschedule.
        '''
        if enable:
            if not self.schedule or not self.timezone:
                raise RuntimeError('Task plan must have schedule and timezone to be enabled')
        self.enabled = enable
        self.update_time = datetime.utcnow()
        MTDB.update('TaskPlan', 'id', self.id, values={'enabled': self.enabled, 'update_time': self.update_time})
        self.__call_scheduler('PUT', f'/task/{self.id}')

    def jobinfo(self) -> dict:
        '''Get scheduled job status of the task plan.

        Returns:
            A dict like {job_id: ..., job_frequency: ..., next_run_time: ...}
        '''
        return self.__call_scheduler('GET', f'/task/{self.id}')

    def save(self) -> None:
        '''Schedule the task plan and save it to database.

        Fail if plan ID or plan name already exists.
        '''
        MTDB.save('TaskPlan', self, on_conflict='update')
        self.__call_scheduler('PUT', f'/task/{self.id}')

    def delete(self) -> None:
        '''Unschedule a task plan and delete it from database.
        '''
        MTDB.delete('TaskPlan', 'id', self.id)
        self.__call_scheduler('DELETE', f'/task/{self.id}')

    def list_logs(self, limit=30) -> list[TaskLog]:
        '''Return all backtest history for this task plan.

        Returns:
            A list of zero or more task logs
        '''
        return MTDB.get_all(
            'TaskLog', 'plan_id', self.id, orderby=('log_time', False),
            limit=limit, cls=TaskLog)

    def get_log(self, run_id: str) -> TaskLog:
        '''Return task log for a particular run.

        Args:
            run_id: The task run ID

        Returns:
            Backtest log, or None if not found
        '''
        return MTDB.get_one('TaskLog', 'id', run_id, cls=TaskLog)


class TaskManager:
    '''TaskManager provides methods to organizes task files and their execution.'''

    __task_root = expanduser('~/.minitrade/task')

    @staticmethod
    def locate(filename: str) -> str:
        ''' Get the absolute path of a task file '''
        return os.path.join(TaskManager.__task_root, filename)

    @staticmethod
    def list() -> list[str]:
        '''Return the list of available task files

        Returns:
            A list of task filenames
        '''
        return [f for f in os.listdir(TaskManager.__task_root) if os.path.isfile(TaskManager.locate(f))]

    @staticmethod
    def save(filename: str, content: str):
        '''Save `content` to a strategy file. 

        Args:
            filename: Strategy file name
            content: File content as string
        '''
        with open(TaskManager.locate(filename), 'w') as f:
            f.write(content)
        plan = TaskPlan.get_plan(filename)
        if plan:
            plan.update_time = datetime.utcnow()
        else:
            plan = TaskPlan(id=MTDB.uniqueid(), name=filename, task_file=filename, timezone=None,
                            schedule=None, notification=None, enabled=False,
                            create_time=datetime.utcnow(), update_time=None)
            with open(TaskManager.locate(filename), 'r') as f:
                line = f.readline()
            try:
                # The first line in file can be a comment that contains default config like:
                # # Default config: {"timezone": "America/New_York", "schedule": "0 9 * * MON-FRI", "enabled": true, "notification": {"telegram": "N", "email": "N"}}
                if line.startswith('# Default config:'):
                    config = json.loads(line[17:])
                    timezone = config.get('timezone')
                    schedule = config.get('schedule')
                    enabled = config.get('enabled', False)
                    notification = config.get('notification')
                    plan = TaskPlan(id=MTDB.uniqueid(), name=filename, task_file=filename, timezone=timezone,
                                    schedule=schedule, notification=notification, enabled=enabled,
                                    create_time=datetime.utcnow(), update_time=None)
            except Exception:
                logger.warning(f'Ignore invalid default config in "{filename}": {line}')
        plan.save()

    @staticmethod
    def read(task_file: str) -> str:
        ''' Return the content of `task_file` 

        Args:
            task_file: The task file name

        Returns:
            The file content as a string
        '''
        with open(TaskManager.locate(task_file), 'r') as f:
            return f.read()

    @staticmethod
    def delete(task_file: str) -> None:
        '''Delete `strategy_file`

        Args:
            task_file: The task file name
        '''
        os.remove(TaskManager.locate(task_file))


class TaskRunner:
    '''TaskRunner handles the execution of a task, write logs, and send notifications.'''

    def __init__(self, plan: TaskPlan):
        self.plan = plan
        self.code = TaskManager.read(self.plan.task_file)

    def execute(self) -> TaskLog:
        '''Run task in an isolated process.

        Returns:
            `TaskLog` of the run
        '''
        try:
            proc = subprocess.run(
                [sys.executable, TaskManager.locate(self.plan.task_file)],
                capture_output=True,
                cwd=TaskManager.locate('.'),
                timeout=600)
            log = TaskLog(
                id=MTDB.uniqueid(),
                plan_id=self.plan.id,
                plan_name=self.plan.name,
                task_code=self.code,
                return_value=proc.returncode,
                stdout=proc.stdout if proc.stdout else None,
                stderr=proc.stderr if proc.stderr else None,
                log_time=datetime.utcnow()
            )
            log.save()
        except Exception:
            ex = traceback.format_exc()
            log = TaskLog(
                id=MTDB.uniqueid(),
                plan_id=self.plan.id,
                plan_name=self.plan.name,
                task_code=self.code,
                return_value=10000,
                stdout=None,
                stderr=ex,
                log_time=datetime.utcnow()
            )
            log.save()
        finally:
            log = self.plan.get_log(log.id)
            self._send_task_notification(log)
            return log

    def _send_task_notification(self, log: TaskLog):
        ''' Send task result via telegram and/or email '''
        telegram = self.plan.notification.get('telegram', 'N') if self.plan.notification else 'N'
        email = self.plan.notification.get('email', 'N') if self.plan.notification else 'N'
        subject = f'Task "{self.plan.name}" {"❌" if log.error else "✅"}  @ {log.log_time.strftime("%Y-%m-%d %H:%M:%S")}'
        message = [f'<b>{subject}</b>\n<pre></pre>']
        message.append(f'<b>Return value</b>\n<pre>{html.escape(str(log.return_value))}</pre>')
        message.append(f'<b>STDERR</b>\n<pre>{html.escape(log.stderr or "None")}</pre>')
        message.append(f'<b>STDOUT</b>\n<pre>{html.escape(log.stdout or "None")}</pre>')
        message = '\n\n'.join(message)
        if telegram == 'A' or (telegram == 'E' and log.error):
            send_telegram_message(html=message)
        if email == 'A' or (email == 'E' and log.return_value != 0):
            mailjet_send_email(subject, message)
