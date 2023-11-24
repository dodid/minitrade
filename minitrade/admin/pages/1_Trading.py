
from dataclasses import asdict
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from tabulate import tabulate

from minitrade.backtest import calculate_trade_stats
from minitrade.broker import Broker, BrokerAccount
from minitrade.datasource import QuoteSource
from minitrade.trader import (BacktestLog, BacktestRunner, StrategyManager,
                              TradePlan)
from minitrade.trader.scheduler import run_trader_after_backtest
from minitrade.utils.mtdb import MTDB

st.set_page_config(page_title='Trading', layout='wide')


def get_market_calendar_and_timezone(data_source: str, ticker_css: str):
    ''' Get market calendar and timezone from tickers '''
    if ticker_css.strip() == '':
        return None, None
    ds = QuoteSource.get_source(data_source)
    calendars = list(set([ds.ticker_calendar(t) for t in ticker_css.split(',')]))
    zones = list(set([ds.ticker_timezone(t) for t in ticker_css.split(',')]))
    if len(calendars) > 1 or len(zones) > 1:
        st.info(
            f'Cross market trading detected. Use "{calendars[0]}" for market calendar, "{zones[0]}" for market timezone.')
    return calendars[0], zones[0]


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


def parse_trade_time_of_day(trade_time_of_day: str) -> str:
    ''' Parse trade time of day from string '''
    if trade_time_of_day:
        ts = []
        for t in trade_time_of_day.split(','):
            t = t.strip()
            if len(t) > 8:
                start_end, interval = t.split('/')
                start, end = start_end.split('-')
                ts.extend(pd.date_range(start, end, freq=interval).time)
            else:
                ts.append(datetime.strptime(t, '%H:%M:%S').time() if len(
                    t.split(':')) == 3 else datetime.strptime(t, '%H:%M').time())
        return ','.join([t.strftime('%H:%M:%S') for t in sorted(ts)])
    else:
        return None


def show_create_trade_plan_form() -> TradePlan | None:
    account = st.selectbox('Select a broker account', BrokerAccount.list(), format_func=lambda b: b.alias)
    strategy_file = st.selectbox('Pick a strategy', StrategyManager.list())
    data_source = st.selectbox('Select a data source', QuoteSource.AVAILABLE_SOURCES)
    ticker_css = st.text_input(
        'Define the asset space (a list of tickers separated by comma without space)', placeholder='e.g. AAPL,GOOG')
    try:
        market_calendar, market_timezone = get_market_calendar_and_timezone(data_source, ticker_css)
    except Exception as e:
        st.error(e)
    backtest_start_date = st.date_input(
        'Pick a backtest start date (run backtest from that date to give enough lead time to calculate indicators)',
        value=datetime.today() - timedelta(days=30))
    trade_start_date = st.date_input(
        'Pick a trade start date (trade signal before that is surpressed)', min_value=datetime.now().date())
    c1, c2 = st.columns([1, 3])
    entry_type = c1.selectbox('Select order entry type', ['TOO', 'TOC', 'TRG'], format_func=lambda x: {
        'TOO': 'Trade on open (TOO)', 'TOC': 'Trade on close (TOC)', 'TRG': 'Trade regular hours (TRG)'}[x])
    trade_time_of_day = c2.text_input(
        'Set when backtest should run (in market local timezone as a list of HH\:MM[\:SS] separated by comma)',
        placeholder='e.g. 9:30, 15:59:30, 9:30:01-16:00/30min')
    try:
        trade_time_of_day = parse_trade_time_of_day(trade_time_of_day)
        if trade_time_of_day:
            st.info(f'{len(trade_time_of_day.split(","))} runs @ {trade_time_of_day.replace(",", ", ")}')
    except Exception as e:
        st.error(e)
    initial_cash = st.number_input('Set the cash amount to invest', value=0)
    initial_holding = st.text_input(
        'Set the preexisting asset positions (number of shares you already own and to be considered in the strategy)',
        placeholder='e.g. AAPL:100,GOOG:100') or None
    name = st.text_input('Name the trade plan')
    strict = st.radio('Select backtest mode', ['Strict', 'Incremental'], index=0) == 'Strict'
    st.info(
        'In strict mode, repeated backtests are expected to produce identical results, and orders '
        'generated from backtests are expected to be executed successfully at all time. This attempts '
        'to trade a strategy as faithfully as possible. However, it may suffer from multiple external '
        'changes, e.g. data change due to dividend and stock split, failed order execution, etc., '
        'in which case, trading will fail to proceed, and manual intervention is required. '
        'In incremental mode, backtests will be run incrementally on the latest data only and in observation of the '
        'actual portfolio status. Strategies must be designed to run incrementally, i.e. not dependent on previous runs. '
        'This mode is more robust to external changes, but performance may deviate from that of the original strategy. '
        'See [Trading](https://dodid.github.io/minitrade/trading/) for more details.')

    tickers = ticker_resolver(account, ticker_css)
    ticker_map = {k: v['id'] if v is not None else None for k, v in tickers.items()}
    if account and None in ticker_map.values():
        st.warning('Tickers are not fully resolved. You can ignore this warning if unresolved tickers are not traded.')
    dryrun = st.button('Save and dry run')

    if dryrun:
        if initial_holding:
            initial_holding = {k.strip(): int(v)
                               for k, v in [x.strip().split(':') for x in initial_holding.split(',')]}
        if len(ticker_css.strip()) == 0 or len(name.strip()) == 0 or not trade_time_of_day:
            st.error('Please do not leave any required field empty.')
        else:
            return TradePlan(
                id=MTDB.uniqueid(),
                name=name,
                strategy_file=strategy_file,
                ticker_css=ticker_css.replace(' ', ''),
                market_calendar=market_calendar,
                market_timezone=market_timezone,
                data_source=data_source,
                backtest_start_date=backtest_start_date.strftime('%Y-%m-%d'),
                trade_start_date=trade_start_date.strftime('%Y-%m-%d'),
                trade_time_of_day=trade_time_of_day,
                entry_type=entry_type,
                broker_account=account.alias if account else None,
                initial_cash=initial_cash,
                initial_holding=initial_holding,
                strict=strict,
                enabled=False,
                create_time=datetime.utcnow(),
                update_time=None,
                broker_ticker_map=ticker_map
            )


def display_run(plan: TradePlan, log: BacktestLog):
    orders = plan.get_orders(log.id)
    log_status = '❌' if log.error else '✅' if orders else '🟢'
    log_time = log.log_time.replace(tzinfo=ZoneInfo('UTC')).astimezone(
        ZoneInfo(plan.market_timezone)).strftime('%Y-%m-%d %H:%M:%S')
    label = f'{log_status} {log_time}' + (f' **{len(orders)} orders**' if orders else '')
    with st.expander(label):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ['Input', 'Data', 'Result', 'Error', 'Log', 'Orders'])
        if log.params:
            tab1.write(log.params)
        tab2.write(log.data)
        if log.result is not None:
            df = log.result
            df = df[~df.index.str.startswith('_')]
            tab3.text(tabulate(df))
        tab4.code(log.exception)
        tab5.caption('Log - stdout')
        if log.stdout:
            tab5.text(log.stdout)
        tab5.caption('Log - stderr')
        if log.stderr:
            tab5.text(log.stderr)
        if orders:
            tab6.write(pd.DataFrame(orders))


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
    plan_lst = TradePlan.list()
    plan_idx = st.sidebar.radio('Trade plan', list(range(len(plan_lst))),
                                format_func=lambda i: plan_lst[i].name)
    plan = plan_lst[int(plan_idx)] if plan_idx is not None else None
    return plan


def run_trade_plan_once(plan: TradePlan) -> None:
    log = run_trader_after_backtest(plan, force=True)
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
        broker = account and Broker.get_broker(account)
        orders = plan.get_orders()
        show_orders(plan, account, broker, orders)

    with tab3:
        if broker and logs:
            show_performance(plan, logs, broker, orders)

    with tab4:
        st.write(asdict(plan))


def show_orders(plan, account, broker, orders):
    for order in orders:
        trade = broker and broker.find_trades(order)
        broker_order = broker and broker.find_order(order)
        order_status = '✅' if trade else '🟢' if order.broker_order_id else '🚫'
        with st.expander(f'{order_status} {order.signal_time} **{order.ticker} {order.side} {abs(order.size)}**'):
            t1, t2, t3 = st.tabs(['Raw order', 'Broker order', 'Trade status'])
            t1.dataframe(pd.DataFrame([asdict(order)]))
            if broker_order:
                t2.dataframe(pd.DataFrame([broker_order]))
            if trade:
                t3.dataframe(pd.DataFrame(trade))
                if account.broker == 'Manual':
                    c1, c2, c3, c4 = t3.columns(4)
                    price = c1.number_input('Price', key=f'{order.id}-price', value=trade[0]['price'] or 0.0)
                    commission = c2.number_input(
                        'Commission', key=f'{order.id}-comm', value=trade[0]['commission'] or 0.0)
                    dt = c3.date_input('Trade time (market local timezone)',
                                       key=f'{order.id}-date', value=trade[0]['trade_time'])
                    tm = c4.time_input('Trade time', key=f'{order.id}-time',
                                       value=trade[0]['trade_time'] or time(9, 30), label_visibility='hidden')
                    if t3.button('Save', key=f'{order.id}-save'):
                        broker.update_trade(trade[0]['id'], price, commission,
                                            datetime.combine(dt, tm, ZoneInfo(plan.market_timezone)))
                        st.warning('Click "Refresh" button to see change')


def show_performance(plan, logs, broker, orders):
    try:
        data = logs[0].data
        trades = broker.format_trades(orders)
        trade_start_date = datetime.strptime(
            plan.trade_start_date, '%Y-%m-%d').replace(tzinfo=ZoneInfo(plan.market_timezone))
        offset = (data.index < trade_start_date).sum()
        # calculate trade stats from trade_start_date
        # data.index may have mixed timezone due to daylight saving time
        trade_df, equity, pnl, commission_rate, cash_available = calculate_trade_stats(
            data.iloc[offset:], plan.initial_cash, trades, plan.initial_holding)
        if len(equity) > 0:
            rr = equity.pct_change()
            sharpe = rr.mean() / rr.std() * 252**0.5
            cum_rr = (equity / equity.iloc[0] - 1) * 100
            cum_rr.name = 'Return rate (%)'
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                label='Portfolio Value', value=f'{equity.iloc[-1]:,.0f}',
                delta=f'{int(equity.iloc[-1]-equity.iloc[-2]):,}' if len(equity) >= 2 else 0)
            c2.metric(label='PnL', value=f'{(equity.iloc[-1]-equity.iloc[0]):,.0f}',
                      delta=f'{cum_rr.iloc[-1]:.2f}%')
            c3.metric(label='Sharpe Ratio', value=f'{sharpe:.2f}')
            c4.metric(label='Commission Rate', value=f'{commission_rate:.3%}')
            if len(cum_rr) >= 2:
                st.caption('Return Rate (%)')
                st.line_chart(cum_rr, height=200)
            st.caption('Profit and Loss')
            st.write(pnl.style.format('{:,.0f}', na_rep=' '))
            st.caption('Cash Available')
            st.write(f'{int(cash_available):,}')
            st.caption('Trades')
            st.write(trade_df.style.format(na_rep=' '))
            st.caption('Orders')
            broker_orders = pd.DataFrame(trades)
            broker_orders['entry_time'] = broker_orders['entry_time'].dt.tz_convert(plan.market_timezone)
            broker_orders['value'] = (broker_orders['entry_price'] * broker_orders['size']).astype(int)
            st.write(broker_orders)
        else:
            st.write('No stats due to insufficient data')
    except Exception as e:
        st.write('No stats due to insufficient data')


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
