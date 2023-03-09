from .fixture import *


def test_get_broker():
    supported = Broker.AVAILABLE_BROKERS
    assert 'IB' in supported
    assert isinstance(supported['IB'], str)


def test_create_ib_broker_account(clean_db):
    username = os.environ.get('IB_TEST_USERNAME')
    password = os.environ.get('IB_TEST_PASSWORD')
    assert username is not None and password is not None
    account = BrokerAccount(alias='pytest_ib_account', broker='IB', mode='Paper', username=username, password=password)
    account.save()
    assert BrokerAccount.get_account('pytest_ib_account') == account


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
