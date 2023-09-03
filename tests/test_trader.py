from .fixture import *


def test_create_tradeplan_using_manual_broker(
        clean_db,
        create_strategies,
        launch_scheduler
):
    assert StrategyManager.load('rotate_buying.py') is not None
    ticker_css = 'AAPL,GOOG,META'
    market_timezone = 'America/New_York'
    account = BrokerAccount.get_account('Manual')
    broker = Broker.get_broker(account=account)
    broker.connect()
    assert broker.is_ready() == True
    broker_ticker_map = broker.resolve_tickers(ticker_css)
    broker_ticker_map = {k: v[0]['id'] for k, v in broker_ticker_map.items()}
    assert broker_ticker_map == {'AAPL': 'AAPL', 'GOOG': 'GOOG', 'META': 'META'}

    # create plan
    plan = TradePlan(
        id=MTDB.uniqueid(),
        name='pytest_manual_tradeplan',
        strategy_file='rotate_buying.py',
        ticker_css='AAPL,GOOG,META',
        market_calendar='NASDAQ',
        market_timezone=market_timezone,
        data_source='Yahoo',
        backtest_start_date='2023-01-01',
        trade_start_date='2023-01-01',
        trade_time_of_day='20:30:00',
        entry_type='TOO',
        broker_account=account.alias,
        initial_cash=10000,
        enabled=False,
        create_time=datetime.utcnow(),
        update_time=None,
        broker_ticker_map=broker_ticker_map
    )
    plan.save()
    assert TradePlan.get_plan('pytest_manual_tradeplan') == plan


def test_manual_trade():
    trader = Trader()
    trader.execute()


def test_run_backtest_using_manual_broker():
    plan = TradePlan.get_plan('pytest_manual_tradeplan')
    runner = BacktestRunner(plan)
    try:
        run_id = MTDB.uniqueid()
        runner.run_backtest(run_id, fail_fast=False)
        log = plan.get_log(run_id)
    except Exception:
        assert False, 'Running backtest failed'
    assert log.error == False
    assert len(plan.get_orders()) > 0


def test_create_tradeplan_using_ib_broker(
        clean_db,
        create_account,
        create_strategies,
        launch_scheduler,
        launch_ibgateway
):
    assert StrategyManager.load('rotate_buying.py') is not None
    ticker_css = 'AAPL,GOOG,META'
    market_timezone = 'America/New_York'
    account = BrokerAccount.get_account('pytest_ib_account')
    broker = Broker.get_broker(account=account)
    broker.connect()
    assert broker.is_ready() == True
    broker_ticker_map = broker.resolve_tickers(ticker_css)
    broker_ticker_map = {k: v[0]['id'] for k, v in broker_ticker_map.items()}
    assert broker_ticker_map == {'AAPL': 265598, 'GOOG': 208813720, 'META': 107113386}

    # create plan
    plan = TradePlan(
        id=MTDB.uniqueid(),
        name='pytest_ib_tradeplan',
        strategy_file='rotate_buying.py',
        ticker_css='AAPL,GOOG,META',
        market_calendar='NASDAQ',
        market_timezone=market_timezone,
        data_source='Yahoo',
        backtest_start_date='2023-01-01',
        trade_start_date='2023-01-01',
        trade_time_of_day='20:30:00',
        entry_type='TOO',
        broker_account=account.alias,
        initial_cash=10000,
        enabled=False,
        create_time=datetime.utcnow(),
        update_time=None,
        broker_ticker_map=broker_ticker_map
    )
    plan.save()
    assert TradePlan.get_plan('pytest_ib_tradeplan') == plan


def test_run_backtest_using_ib_broker():
    plan = TradePlan.get_plan('pytest_ib_tradeplan')
    runner = BacktestRunner(plan)
    try:
        run_id = MTDB.uniqueid()
        runner.run_backtest(run_id, fail_fast=False)
        log = plan.get_log(run_id)
    except Exception:
        assert False, 'Running backtest failed'
    assert log.error == False
    assert len(plan.get_orders()) > 0


def test_ib_trade(launch_ibgateway):
    trader = Trader()
    trader.execute()
