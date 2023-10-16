from .fixture import *


def test_invalid_plan(clean_db, create_strategies, launch_scheduler):
    plan = TradePlan(
        id=MTDB.uniqueid(),
        name='pytest_run_backtest',
        strategy_file='invalid_strategy.py',
        ticker_css='AAPL,GOOG,META',
        market_calendar='NASDAQ',
        market_timezone='America/New_York',
        data_source='Yahoo',
        backtest_start_date='2023-01-01',
        trade_start_date='2023-01-01',
        trade_time_of_day='20:30:00',
        entry_type='TOO',
        broker_account='Manual',
        initial_cash=10000,
        enabled=True,
        strict=False,
        create_time=datetime.utcnow(),
        update_time=None,
        broker_ticker_map={'AAPL': 'AAPL', 'GOOG': 'GOOG', 'META': 'META'}
    )
    plan.save()
    runner = BacktestRunner(plan)

    # Test invalid plan
    result = runner.run_backtest()
    assert result is None


def test_run_backtest_strict(clean_db, create_strategies, launch_scheduler):
    plan = TradePlan(
        id=MTDB.uniqueid(),
        name='pytest_run_backtest',
        strategy_file='rotate_buying.py',
        ticker_css='AAPL,GOOG,META',
        market_calendar='NASDAQ',
        market_timezone='America/New_York',
        data_source='Yahoo',
        backtest_start_date='2023-01-01',
        trade_start_date='2023-01-01',
        trade_time_of_day='20:30:00',
        entry_type='TRG',
        broker_account='Manual',
        initial_cash=10000,
        enabled=True,
        strict=True,
        create_time=datetime.utcnow(),
        update_time=None,
        broker_ticker_map={'AAPL': 'AAPL', 'GOOG': 'GOOG', 'META': 'META'}
    )
    plan.save()

    runner = BacktestRunner(plan)

    # Test dryrun mode
    run_id = MTDB.uniqueid()
    assert runner.run_backtest(run_id=run_id, dryrun=True) is not None
    assert plan.get_log(run_id).error == False
    assert not plan.get_orders()

    # Test strict mode
    run_id = MTDB.uniqueid()
    assert runner.run_backtest(run_id=run_id) is not None
    assert plan.get_log(run_id).error == False
    assert len(plan.get_orders(run_id=run_id)) > 0

    # Test runs are repeatable in strict mode
    run_id = MTDB.uniqueid()
    assert runner.run_backtest(run_id=run_id) is not None
    assert plan.get_log(run_id).error == False
    assert len(plan.get_orders(run_id=run_id)) == 0

    # Test submit orders
    broker = Broker.get_broker(BrokerAccount.get_account(plan.broker_account))
    assert broker.is_ready()
    assert broker.get_account_info()
    assert not broker.get_portfolio()
    assert not broker.download_trades()
    assert not broker.download_orders()
    Trader().execute()
    assert len(broker.download_trades()) == len(plan.get_orders())


def test_run_backtest_incremental(clean_db, create_strategies, launch_scheduler):
    plan = TradePlan(
        id=MTDB.uniqueid(),
        name='pytest_run_backtest',
        strategy_file='rotate_buying.py',
        ticker_css='AAPL,GOOG,META',
        market_calendar='NASDAQ',
        market_timezone='America/New_York',
        data_source='Yahoo',
        backtest_start_date='2023-01-01',
        trade_start_date='2023-01-01',
        trade_time_of_day='20:30:00',
        entry_type='TRG',
        broker_account='Manual',
        initial_cash=10000,
        enabled=True,
        strict=False,
        create_time=datetime.utcnow(),
        update_time=None,
        broker_ticker_map={'AAPL': 'AAPL', 'GOOG': 'GOOG', 'META': 'META'}
    )
    plan.save()

    runner = BacktestRunner(plan)

    # Test dryrun mode
    run_id = MTDB.uniqueid()
    assert runner.run_backtest(run_id=run_id, dryrun=True) is not None
    assert plan.get_log(run_id).error == False
    assert not plan.get_orders()

    # Test incremental mode
    run_id = MTDB.uniqueid()
    assert runner.run_backtest(run_id=run_id) is not None
    assert plan.get_log(run_id).error == False
    assert len(plan.get_orders(run_id=run_id)) == 1
    prev_run_id = run_id

    # Test runs are incremental
    run_id = MTDB.uniqueid()
    assert runner.run_backtest(run_id=run_id) is not None
    assert plan.get_log(run_id).error == False
    assert len(plan.get_orders(run_id=run_id)) == 1
    assert plan.get_orders(run_id=prev_run_id)[0].cancelled == True

    # Test submit orders
    broker = Broker.get_broker(BrokerAccount.get_account(plan.broker_account))
    Trader().execute()
    assert len(broker.download_trades()) == 1


def test_backtest_storage(clean_db, create_strategies, launch_scheduler):
    plan = TradePlan(
        id=MTDB.uniqueid(),
        name='pytest_run_backtest',
        strategy_file='storage_test.py',
        ticker_css='AAPL,GOOG,META',
        market_calendar='NASDAQ',
        market_timezone='America/New_York',
        data_source='Yahoo',
        backtest_start_date='2023-01-01',
        trade_start_date='2023-01-01',
        trade_time_of_day='20:30:00',
        entry_type='TRG',
        broker_account='Manual',
        initial_cash=10000,
        enabled=True,
        strict=False,
        create_time=datetime.utcnow(),
        update_time=None,
        broker_ticker_map={'AAPL': 'AAPL', 'GOOG': 'GOOG', 'META': 'META'}
    )
    plan.save()

    plan = TradePlan.get_plan(plan.name)
    assert isinstance(plan.storage, dict)
    assert len(plan.storage) == 0

    # Dryrun mode doesn't change storage
    assert BacktestRunner(TradePlan.get_plan(plan.name)).run_backtest(dryrun=True) is not None
    assert len(TradePlan.get_plan(plan.name).storage) == 0

    # Storage is initialized correctly
    assert BacktestRunner(TradePlan.get_plan(plan.name)).run_backtest() is not None
    storage = TradePlan.get_plan(plan.name).storage
    assert storage['int'] == 0
    assert storage['float'] == 0.0
    assert storage['str'] == '0'
    assert storage['list'] == [0]
    assert storage['dict'] == {'int': 0}
    assert np.array_equal(storage['np'], np.array([0]))
    assert storage['pd'].equals(pd.DataFrame({'int': [0]}))

    # Storage is updated correctly
    assert BacktestRunner(TradePlan.get_plan(plan.name)).run_backtest() is not None
    storage = TradePlan.get_plan(plan.name).storage
    assert storage['int'] == 1
    assert storage['float'] == 1.0
    assert storage['str'] == '1'
    assert storage['list'] == [1]
    assert storage['dict'] == {'int': 1}
    assert np.array_equal(storage['np'], np.array([1]))
    assert storage['pd'].equals(pd.DataFrame({'int': [1]}))
