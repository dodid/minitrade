import os
import tempfile

import pandas as pd
import quantstats as qs
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


def show_ib_account_overview(tab, info, portfolio):
    with tab:
        account = st.selectbox(
            'Account', info, format_func=lambda x: f'{x["displayName"]} ({x["accountId"]})', key='ib_account_overview')
        if 'ledger' in account:
            ledger = account['ledger']['BASE']
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric('Net Liquidation', f'{ledger.get("netliquidationvalue", 0):,.0f}')
            c2.metric('Cash Balance', f'{ledger.get("cashbalance", 0):,.0f}')
            c3.metric('Interest', f'{ledger.get("interest", 0):,.0f}')
            c4.metric('Dividends', f'{ledger.get("dividends", 0):,.0f}')
            c5.metric('Unrealized PnL', f'{ledger.get("unrealizedpnl", 0):,.0f}')
            c6.metric('Realized PnL', f'{ledger.get("realizedpnl", 0):,.0f}')
            c1.metric('Stock Value', f'{ledger.get("stockmarketvalue", 0):,.0f}')
            c2.metric('Option Value', f'{ledger.get("stockoptionmarketvalue", 0):,.0f}')
            c3.metric('Future Value', f'{ledger.get("futuremarketvalue", 0):,.0f}')
            c4.metric('Future Option Value', f'{ledger.get("futureoptionmarketvalue", 0):,.0f}')
            c5.metric('Warrant Value', f'{ledger.get("warrantsmarketvalue", 0):,.0f}')
            c6.metric('Commodity Value', f'{ledger.get("commoditymarketvalue", 0):,.0f}')
            c1.metric('Bond Value', f'{ledger.get("tbondsmarketvalue", 0):,.0f}')
            c2.metric('Bill Value', f'{ledger.get("tbillsmarketvalue", 0):,.0f}')
            c3.metric('Corporate Bond Value', f'{ledger.get("corporatebondsmarketvalue", 0):,.0f}')
            c4.metric('Fund Value', f'{ledger.get("funds", 0):,.0f}')
            c5.metric('Money Fund Value', f'{ledger.get("moneyfunds", 0):,.0f}')
            c6.metric('Crypto Value', f'{ledger.get("cryptocurrencyvalue", 0):,.0f}')
        if 'performance' in account:
            c1, c2 = st.columns(2)
            c1.caption('NAV')
            nav = account['performance']['nav']
            nav_s = pd.Series(nav['data'][0]['navs'], index=pd.to_datetime(nav['dates'], format='%Y%m%d'))
            c1.line_chart(nav_s, height=300)
            c2.caption('Cumulative PnL')
            cps = account['performance']['cps']
            cps_s = pd.Series(cps['data'][0]['returns'], index=pd.to_datetime(cps['dates'], format='%Y%m%d'))
            c2.line_chart(cps_s, height=300)
        st.caption('Portfolio')
        st.write(portfolio[portfolio['acctId'] == account['id']])


def show_ib_tearsheet(tab, info):
    with tab:
        account = st.selectbox(
            'Account', info, format_func=lambda x: f'{x["displayName"]} ({x["accountId"]})', key='ib_tearsheet')
        if 'performance' in account:
            cps = account['performance']['cps']
            cps_s = pd.Series(cps['data'][0]['returns'], index=pd.to_datetime(cps['dates'], format='%Y%m%d'))
            temp = os.path.join(tempfile.gettempdir(), 'quantstats-tearsheet.html')
            qs.reports.html((cps_s+1).pct_change().dropna(), benchmark='SPY', rf=0.0, display=False,
                            output=temp, title=f'{account["displayName"]} ({account["accountId"]})')
            with open(temp) as f:
                st.components.v1.html(f.read(), height=6000)


def show_broker_account_portfolio_and_trades(account: BrokerAccount):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Overview', 'Recent trades', 'Orders', 'Account info', 'Performance'])
    try:
        info, portfolio, trades, orders = get_account_info(account)
        if account.broker == 'IB':
            show_ib_account_overview(tab1, info, portfolio)
            show_ib_tearsheet(tab5, info)
        else:
            tab1.caption('Portfolio')
            tab1.write(portfolio)
            tab5.write('Not available for this account')
        tab2.write(trades)
        tab3.write(orders)
        tab4.write(info)
    except ConnectionError:
        st.info('Login to see account info')


@st.cache_data(ttl='1h')
def get_account_info(account):
    broker = Broker.get_broker(account)
    if broker.is_ready():
        return broker.get_account_info(), broker.get_portfolio(), broker.download_trades(), broker.download_orders()
    else:
        raise ConnectionError('Broker is not ready')


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
