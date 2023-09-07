from zoneinfo import ZoneInfo

from minitrade.broker.ib import InteractiveBrokersValidator

from .fixture import *


def test_get_broker():
    supported = Broker.AVAILABLE_BROKERS
    assert 'IB' in supported
    assert isinstance(supported['IB'], str)
    assert 'Manual' in supported
    assert isinstance(supported['Manual'], str)


def test_builtin_manual_broker_account(clean_db):
    assert BrokerAccount.get_account('Manual') is not None


def test_create_manual_broker_account(clean_db):
    account = BrokerAccount(alias='pytest_manual_account', broker='Manual',
                            mode='Paper', username='username', password='password')
    account.save()
    assert BrokerAccount.get_account('pytest_manual_account') == account


def test_order_validator():
    plan = TradePlan(
        id='valid',
        name='test',
        strategy_file='test.py',
        ticker_css='valid',
        market_calendar='SSE',
        market_timezone='Asia/Shanghai',
        data_source='EastMoney',
        backtest_start_date='2023-01-01',
        trade_start_date=None,
        trade_time_of_day='19:00',
        entry_type='TOO',
        broker_account='pytest_manual_account',
        commission_rate=0.001,
        initial_cash=10000,
        initial_holding=None,
        enabled=True,
        create_time=datetime.now(),
        update_time=None,
        broker_ticker_map={'valid': 'valid'}
    )
    validator = OrderValidator(plan)
    order = RawOrder(id='invalid', plan_id='invalid', run_id='invalid', ticker='invalid', side='invalid',
                     size=0, signal_time=datetime.now(), broker_order_id=1)
    with pytest.raises(AttributeError):
        validator.order_has_correct_plan_id(order)
    with pytest.raises(AttributeError):
        validator.order_has_correct_run_id(order)
    with pytest.raises(AttributeError):
        validator.order_has_correct_ticker(order)
    with pytest.raises(AttributeError):
        validator.order_has_correct_size(order)
    with pytest.raises(AttributeError):
        validator.order_has_no_broker_order_id(order)
    with pytest.raises(AttributeError):
        validator.order_is_in_sync_with_db(order)
    with pytest.raises(RuntimeError):
        validator.validate(order)

    run_log = BacktestLog(
        id='valid', plan_id=plan.id, plan_name=plan.name, plan=plan, plan_strategy=plan.strategy_file, data=None,
        strategy_code=None, result=None, exception=None, stdout=None, stderr=None, log_time=datetime.now(), params=None)
    run_log.save()

    order = RawOrder(id='valid', plan_id='valid', run_id='valid', ticker='valid', side='Buy',
                     size=100, signal_time=datetime.now(), broker_order_id=None)
    order.save()
    validator.validate(order)

    order = RawOrder(id='valid2', plan_id='valid', run_id='valid', ticker='valid', side='Sell',
                     size=-100, signal_time=datetime.now(), broker_order_id=None)
    order.save()
    validator.validate(order)


def test_manual_broker_works():
    account = BrokerAccount.get_account('pytest_manual_account')
    broker = Broker.get_broker(account=account)
    assert broker.is_ready() == True
    try:
        broker.connect()
    except Exception:
        assert False, f'Broker not connected'
    assert broker.is_ready() == True
    try:
        broker.get_account_info()
        broker.get_portfolio()
        broker.download_trades()
        broker.download_orders()
    except Exception as e:
        assert False, f'Unexpected exception {e}'


def test_delete_manual_broker_account():
    account = BrokerAccount.get_account('pytest_manual_account')
    assert account.alias == 'pytest_manual_account'
    account.delete()
    assert BrokerAccount.get_account('pytest_manual_account') is None


def test_create_ib_broker_account(clean_db):
    username = os.environ.get('IB_TEST_USERNAME')
    password = os.environ.get('IB_TEST_PASSWORD')
    assert username is not None and password is not None
    account = BrokerAccount(alias='pytest_ib_account', broker='IB', mode='Paper', username=username, password=password)
    account.save()
    assert BrokerAccount.get_account('pytest_ib_account') == account


def test_ib_order_validator():
    account = BrokerAccount.get_account('pytest_ib_account')
    broker = Broker.get_broker(account=account)
    plan = TradePlan(
        id='valid',
        name='test',
        strategy_file='test.py',
        ticker_css='AAPL',
        market_calendar='NASDAQ',
        market_timezone='America/New_York',
        data_source='Yahoo',
        backtest_start_date='2023-01-01',
        trade_start_date=None,
        trade_time_of_day='19:00',
        entry_type='TOO',
        broker_account='pytest_ib_account',
        commission_rate=0.001,
        initial_cash=10000,
        initial_holding=None,
        enabled=True,
        create_time=datetime.now(),
        update_time=None,
        broker_ticker_map={'AAPL': 265598}
    )
    MTDB.save(plan, 'TradePlan')
    validator = InteractiveBrokersValidator(plan, broker, pytest_now=datetime(
        2023, 1, 3, 18, 0, tzinfo=ZoneInfo(plan.market_timezone)))
    order = RawOrder(id='invalid', plan_id='invalid', run_id='invalid', ticker='invalid', side='invalid',
                     size=10001, signal_time=datetime.now(), broker_order_id=1)
    with pytest.raises(AttributeError):
        validator.order_has_correct_plan_id(order)
    with pytest.raises(AttributeError):
        validator.order_has_correct_run_id(order)
    with pytest.raises(AttributeError):
        validator.order_has_correct_ticker(order)
    with pytest.raises(AttributeError):
        validator.order_has_correct_size(order)
    with pytest.raises(AttributeError):
        validator.order_has_no_broker_order_id(order)
    with pytest.raises(AttributeError):
        validator.order_is_in_sync_with_db(order)
    with pytest.raises(AttributeError):
        validator.order_in_time_window(order)
    with pytest.raises(AttributeError):
        MTDB.save({'execution_id': 'an_id', 'order_ref': order.id}, 'IbTrade', on_conflict='update')
        validator.order_not_in_finished_trades(order)
    with pytest.raises(AttributeError):
        MTDB.save({'orderId': 'an_id', 'order_ref': order.id}, 'IbOrder', on_conflict='update')
        validator.order_not_in_open_orders(order)
    with pytest.raises(AttributeError):
        validator.order_size_is_within_limit(order)
    with pytest.raises(RuntimeError):
        validator.validate(order)

    run_log = BacktestLog(
        id='valid', plan_id=plan.id, plan_name=plan.name, plan=plan, plan_strategy=plan.strategy_file, data=None,
        strategy_code=None, result=None, exception=None, stdout=None, stderr=None, log_time=datetime.now(), params=None)
    run_log.save()

    order = RawOrder(id='valid', plan_id='valid', run_id='valid', ticker='AAPL', side='Buy', size=100,
                     signal_time=datetime(2023, 1, 3, tzinfo=ZoneInfo(plan.market_timezone)), broker_order_id=None)
    order.save()
    validator.validate(order)

    order = RawOrder(id='valid', plan_id='valid', run_id='valid', ticker='AAPL', side='Buy', size=100,
                     signal_time=datetime(2023, 1, 2, tzinfo=ZoneInfo(plan.market_timezone)), broker_order_id=None)
    order.save()
    with pytest.raises(RuntimeError):
        validator.validate(order)


def test_ib_broker_works(launch_ibgateway):
    # Got this error when running on mac. Use the following fix
    # https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr
    # export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    account = BrokerAccount.get_account('pytest_ib_account')
    broker = Broker.get_broker(account=account)
    assert broker.is_ready() == False
    try:
        broker.connect()
    except Exception:
        assert False, f'Broker not connected'
    assert broker.is_ready() == True
    try:
        broker.get_account_info()
        broker.get_portfolio()
        broker.download_trades()
        broker.download_orders()
    except Exception as e:
        assert False, f'Unexpected exception {e}'


def test_delete_ib_broker_account():
    account = BrokerAccount.get_account('pytest_ib_account')
    assert account.alias == 'pytest_ib_account'
    account.delete()
    assert BrokerAccount.get_account('pytest_ib_account') is None
