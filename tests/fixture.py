import os
import pkgutil
import sqlite3
import time
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from posixpath import expanduser

import numpy as np
import pandas as pd
import pytest

from minitrade.backtest import *
from minitrade.broker import *
from minitrade.datasource import *
from minitrade.trader import *
from minitrade.utils.mtdb import MTDB


def populate_test_tickers():
    tickers = [
        ('AAPL', 'Apple Inc.', 'NASDAQ', 'America/New_York', ''),
        ('GOOG', 'Alphabet Inc.', 'NASDAQ', 'America/New_York', ''),
        ('META', 'Meta Platforms', 'NASDAQ', 'America/New_York', ''),
        ('SPY', 'SPDR S&P 500 ETF Trust', 'NASDAQ', 'America/New_York', ''),
        ('0000001', 'Ping An', 'SSE', 'Asia/Shanghai', 'SH'),
        ('0000002', 'China Vanke', 'SSE', 'Asia/Shanghai', 'SH'),
        ('0000003', 'China Construction Bank', 'SSE', 'Asia/Shanghai', 'SH'),
    ]
    df = pd.DataFrame(tickers, columns=['ticker', 'name', 'calendar', 'timezone', 'yahoo_modifier'])
    df.to_sql('Ticker', MTDB.conn(), if_exists='replace', index=False)


@pytest.fixture
def clean_db():
    '''Recreate all tables in minitrade.pytest.db'''
    db_loc = expanduser('~/.minitrade/database/minitrade.pytest.db')
    sql = pkgutil.get_data('minitrade.cli', 'minitrade.db.sql').decode('utf-8')
    with sqlite3.connect(db_loc) as conn:
        conn.executescript(sql)
    conn.close()
    populate_test_tickers()


@pytest.fixture
def clean_strategy():
    '''Delete all files in stategy directory'''
    st_dir = expanduser('~/.minitrade/strategy')
    for file in os.listdir(st_dir):
        st_loc = os.path.join(st_dir, file)
        if st_loc.endswith('.py'):
            Path(st_loc).unlink(missing_ok=True)


def ib_start():
    import uvicorn

    from minitrade.utils.config import config
    uvicorn.run(
        'minitrade.broker.ibgateway:app',
        host=config.brokers.ib.gateway_admin_host,
        port=config.brokers.ib.gateway_admin_port,
        log_level=config.brokers.ib.gateway_admin_log_level,
    )


def scheduler_start():
    import uvicorn

    from minitrade.utils.config import config
    uvicorn.run(
        'minitrade.trader.scheduler:app',
        host=config.scheduler.host,
        port=config.scheduler.port,
        log_level=config.scheduler.log_level,
    )


@pytest.fixture
def launch_scheduler():
    '''Launch and shutdown scheduler'''
    schd = Process(target=scheduler_start)
    schd.start()
    time.sleep(5)
    yield
    schd.terminate()


@pytest.fixture
def launch_ibgateway():
    ''' Launch and shutdown IB gateway'''
    gateway = Process(target=ib_start)
    gateway.start()
    time.sleep(3)
    yield
    gateway.terminate()


@pytest.fixture
def create_account():
    username = os.environ.get('IB_TEST_USERNAME')
    password = os.environ.get('IB_TEST_PASSWORD')
    assert username is not None and password is not None
    account = BrokerAccount(alias='pytest_ib_account', broker='IB', mode='Paper', username=username, password=password)
    account.save()
    assert BrokerAccount.get_account('pytest_ib_account') == account


@pytest.fixture
def create_strategies():
    st_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'strategy')
    for file in os.listdir(st_dir):
        if not file.endswith('.py'):
            continue
        with open(os.path.join(st_dir, file), 'r') as f:
            content = f.read()
        StrategyManager.save(file, content)
        assert file in StrategyManager.list()
        strategy = StrategyManager.load(file)
        assert issubclass(strategy, Strategy)
