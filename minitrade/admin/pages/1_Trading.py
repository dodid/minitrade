
from dataclasses import asdict
from datetime import datetime, time, timedelta

import pandas as pd
import streamlit as st

from minitrade.backtest import calculate_trade_stats
from minitrade.broker import Broker, BrokerAccount
from minitrade.datasource import QuoteSource
from minitrade.trader import (BacktestLog, BacktestRunner, StrategyManager,
                              TradePlan)
from minitrade.utils.mtdb import MTDB

st.set_page_config(page_title='Trading', layout='wide')


def market_timezone_selectbox():
    major_market_timezones = ['America/New_York', 'Asia/Shanghai', 'Asia/Hong_Kong']
    return st.selectbox('Select market local timezone', options=major_market_timezones)


def ticker_resolver(account: BrokerAccount, ticker_css: str) -> dict:
    ''' Map ticker to broker specific contract ID '''
    if account and len(ticker_css.replace(' ', '')) > 0:
        broker = Broker.get_broker(account)
        with st.expander('Resolve tickers', expanded=True):
            try:
                broker.connect()
                candidates: dict = broker.resolve_tickers(ticker_css.replace(' ', ''))
                if candidates:
                    vendor_tickers = {}
                    for k, v in candidates.items():
                        vendor_tickers[k] = st.radio(k, v, format_func=lambda _: _['label'])
                    return vendor_tickers
            except ConnectionError:
                st.error('Need to login to broker account to resolve tickers. Please pay attention to 2FA notification if enabled.')
                st.button('Retry')
    return None


def get_broker_ticker_map(tickers: dict | None) -> dict | None:
    ''' Verify if all tickers have been resolved and return the mapping '''
    if tickers and None not in tickers.values():
        return {k: v['id'] for k, v in tickers.items()}
    else:
        return None


def show_create_trade_plan_form() -> TradePlan | None:
    strategy_file = st.selectbox('Pick a strategy', StrategyManager.list())
    ticker_css = st.text_input('Define the asset space')
    data_source = st.selectbox('Select a data source', QuoteSource.AVAILABLE_SOURCES)
    market_timezone = market_timezone_selectbox()
    backtest_start_date = st.date_input(
        'Pick a backtest start date (run backtest from that date)', value=datetime.today()-timedelta(days=110))
    trade_start_date = st.date_input(
        'Pick a trade start date (trade signal before that is surpressed)', min_value=datetime.now().date())
    trade_time_of_day = st.time_input('Pick when backtesting should start (market local time)', value=time(20, 30))
    account = st.selectbox('Select a broker account', BrokerAccount.list(), format_func=lambda b: b.alias)
    initial_cash = st.number_input('Set the cash amount to invest', value=10000)
    name = st.text_input('Name the trade plan')
    tickers = ticker_resolver(account, ticker_css)
    dryrun = st.button('Save and dry run')
    if dryrun:
        ticker_map = get_broker_ticker_map(tickers)
        if len(ticker_css.strip()) == 0 or len(name.strip()) == 0:
            st.error('Please do not leave any field empty.')
        elif account and ticker_map is None:
            st.error('Tickers are not fully resolved.')
        else:
            return TradePlan(
                id=MTDB.uniqueid(),
                name=name,
                strategy_file=strategy_file,
                ticker_css=ticker_css.replace(' ', ''),
                market_timezone=market_timezone,
                data_source=data_source,
                backtest_start_date=backtest_start_date.strftime('%Y-%m-%d'),
                trade_start_date=trade_start_date.strftime('%Y-%m-%d'),
                trade_time_of_day=trade_time_of_day.strftime('%H:%M:%S'),
                broker_account=account.alias if account else None,
                initial_cash=initial_cash,
                enabled=False,
                create_time=datetime.utcnow(),
                update_time=None,
                broker_ticker_map=ticker_map
            )


def display_run(plan: TradePlan, log: BacktestLog):
    orders = plan.get_orders(log.id)
    log_status = '???' if log.error else '???' if orders else '????'
    label = f'{log_status} {log.log_time} [{log.id}]' + (f' **{len(orders)} orders**' if orders else '')
    with st.expander(label):
        tab1, tab2, tab3, tab4 = st.tabs(['Result', 'Error', 'Log', 'Orders'])
        if log.result is not None:
            df = log.result
            df = df[~df.index.str.startswith('_')]['0'].to_dict()
            tab1.write(df)
        tab2.code(log.exception)
        tab3.caption('Log - stdout')
        if log.stdout:
            tab3.text(log.stdout)
        tab3.caption('Log - stderr')
        if log.stderr:
            tab3.text(log.stderr)
        if orders:
            tab4.write(pd.DataFrame(orders))


def save_plan_and_dryrun(plan: TradePlan) -> None:
    plan.save()
    runner = BacktestRunner(plan)
    log = runner.execute(dryrun=True)
    display_run(plan, log)
    if log.error:
        st.error('Trade plan dryrun failed, plan disabled')
    else:
        plan.enable(True)
        st.success('Trade plan dryrun succeeded, plan enabled')


def show_trade_plan_selector() -> TradePlan | None:
    plan_lst = TradePlan.list_plans()
    plan_idx = st.sidebar.radio('Trade plan', list(range(len(plan_lst))),
                                format_func=lambda i: plan_lst[i].name)
    plan = plan_lst[int(plan_idx)] if plan_idx is not None else None
    return plan


def run_trade_plan_once(plan: TradePlan) -> None:
    runner = BacktestRunner(plan)
    log = runner.execute()
    if log is not None and not log.error:
        st.success(f'Backtest {log.id} finished successfully')
    else:
        st.error(f'Backtest failed')


def confirm_delete_trade_plan(plan: TradePlan) -> None:
    def confirm_delete():
        if st.session_state.delete_confirm_textinput == plan.name:
            plan.delete()
    st.text_input(f'Type "{plan.name}" and press Enter to delete',
                  on_change=confirm_delete, key='delete_confirm_textinput')


def show_trade_plan_header_and_controls(plan: TradePlan) -> None:
    c1, c2, c3, c4, c5 = st.columns([5, 1, 1, 1, 1])
    c1.subheader(plan.name)
    c2.button('Refresh')
    c3.button('Run once', on_click=lambda: run_trade_plan_once(plan))
    c4.button('Disable' if plan.enabled else 'Enable', key='tradeplan_onoff',
              on_click=lambda: plan.enable(not plan.enabled))
    if c5.button('Delete', type='primary'):
        confirm_delete_trade_plan(plan)


def show_trade_plan_status(plan: TradePlan) -> None:
    job = plan.jobinfo()
    if job is None:
        if plan.enabled:
            st.warning('State inconsistency detected: plan is enabled but not scheduled')
        else:
            st.warning('Plan is disabled and not trading')
    else:
        if plan.enabled:
            st.success(f'Plan is scheduled to run at {job["next_run_time"]}')
        else:
            st.success(
                f'State inconsistenccy detected: plan is disabled but is scheduled to run at {job["next_run_time"]}')


def show_trade_plan_execution_history(plan: TradePlan) -> None:
    tab1, tab2, tab3, tab4 = st.tabs(['Run history', 'Orders', 'Performance', 'Settings'])
    logs = plan.list_logs()
    with tab1:
        show_trade_plan_status(plan)
        for log in logs:
            display_run(plan, log)

    with tab2:
        account = BrokerAccount.get_account(plan)
        broker = Broker.get_broker(account)
        orders = plan.get_orders()
        for order in orders:
            trade = broker.find_trades(order)
            broker_order = broker.find_order(order)
            order_status = '???' if trade else '????' if order.broker_order_id else '????'
            with st.expander(f'{order_status} {order.signal_time} **{order.ticker} {order.side} {abs(order.size)}**'):
                st.caption('Raw order')
                st.write(asdict(order))
                if broker_order:
                    st.caption('Broker order')
                    st.write(broker_order)
                if trade:
                    st.caption('Trade status')
                    st.write(trade)

    with tab3:
        try:
            data = logs[0].data
            trades = broker.format_trades(orders)
            _, trade_df, equity, pnl, commission_rate = calculate_trade_stats(data, plan.initial_cash, trades)
            rr = (equity / equity[0] - 1) * 100
            rr.name = 'Return rate (%)'
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(label='Portfolio Value',
                      value=f'{equity.iloc[-1]:,.0f}', delta=f'{int(equity.iloc[-1]-equity.iloc[-2]):,}')
            c2.metric(label='PnL', value=f'{(equity.iloc[-1]-plan.initial_cash):,.0f}')
            c3.metric(label='Return', value=f'{rr.iloc[-1]:.2f}%')
            c4.metric(label='Commission Rate', value=f'{commission_rate:.3%}')
            st.caption('Return rate (%)')
            st.line_chart(rr, height=200)
            st.caption('Profit and loss')
            st.write(pnl.style.format('{:,.0f}', na_rep=' '))
            st.caption('Trades')
            st.write(trade_df.style.format(na_rep=' '))
        except Exception as e:
            st.write(e)

    with tab4:
        st.write(asdict(plan))


action = st.sidebar.radio('Action', ['Browse existing', 'Create new'])

if action == 'Create new':
    st.subheader('Create new trade plan')
    plan = show_create_trade_plan_form()
    if plan:
        save_plan_and_dryrun(plan)

if action == 'Browse existing':
    plan = show_trade_plan_selector()
    if plan:
        show_trade_plan_header_and_controls(plan)
        show_trade_plan_execution_history(plan)
