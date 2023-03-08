import streamlit as st

from minitrade.broker import Broker, BrokerAccount

st.set_page_config(page_title='Broker', layout='wide')


def show_create_broker_account_form():
    brokers = Broker.AVAILABLE_BROKERS
    broker = st.selectbox('Broker', brokers.keys(), format_func=lambda k: brokers[k])
    mode = st.radio('Account type', ['Paper', 'Live'])
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    alias = st.text_input('Alias', value=f'{broker}-{mode}-{username}')
    save = st.button('Test and save')
    if save:
        return BrokerAccount(
            alias=alias,
            broker=broker,
            mode=mode,
            username=username,
            password=password,
        )


def test_and_save_broker_account(account: BrokerAccount):
    if account:
        account.save()
        broker = Broker.get_broker(account)
        st.info(
            f'Logging in account {account.username}. Please pay attention to 2FA notification if enabled.')
        try:
            broker.connect()
            st.success('Login succeeded')
        except ConnectionError:
            st.error('Login failed')


def show_broker_account_selector():
    acc_lst = BrokerAccount.list()
    acc_idx = st.sidebar.radio('Broker account', list(range(len(acc_lst))),
                               format_func=lambda i: acc_lst[i].alias)
    account = acc_lst[int(acc_idx)] if acc_idx is not None else None
    return account


def confirm_delete_broker_account(account: BrokerAccount) -> None:
    def confirm_delete():
        if st.session_state.delete_confirm_textinput == account.alias:
            account.delete()
    st.text_input(f'Type "{account.alias}" and press Enter to delete',
                  on_change=confirm_delete, key='delete_confirm_textinput')


def show_broker_account_header_and_controls(account: BrokerAccount):
    c1, c2, c3, c4 = st.columns([6, 1, 1, 1])
    c1.subheader(account.alias)
    c2.button('Refresh')
    if c3.button('Login'):
        broker = Broker.get_broker(account)
        try:
            broker.connect()
            st.success('Login succeeded')
        except ConnectionError:
            st.error('Login failed')
    if c4.button('Delete', type='primary'):
        confirm_delete_broker_account(account)


def show_broker_account_portfolio_and_trades(account: BrokerAccount):
    tab1, tab2, tab3, tab4 = st.tabs(['Portfolio', 'Recent trades', 'Orders', 'Account info'])
    broker = Broker.get_broker(account)
    if broker.is_ready():
        info = broker.get_account_info()
        tab1.write(broker.get_portfolio())
        tab2.write(broker.download_trades())
        tab3.write(broker.download_orders())
        tab4.write(info)
    else:
        st.info('Login to see account info')


action = st.sidebar.radio('Action', ['Browse existing', 'Add new'])

if action == 'Add new':
    st.subheader('Add broker account')
    account = show_create_broker_account_form()
    if account:
        test_and_save_broker_account(account)

if action == 'Browse existing':
    account = show_broker_account_selector()
    if account:
        show_broker_account_header_and_controls(account)
        show_broker_account_portfolio_and_trades(account)
