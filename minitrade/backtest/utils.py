'''
`minitrade.backtest.utils` provides some helper functions for strategy research.
'''

import itertools
import random
from math import copysign
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from minitrade.backtest.core import Backtest, Strategy

plt.rcParams["figure.figsize"] = (20, 3)
plt.rcParams['axes.grid'] = True

matplotlib.rcParams['font.family'] = ['Heiti TC']

try:
    from IPython.display import display
except ModuleNotFoundError:
    display = print

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx():
        import streamlit as st
        print = display = st.write
except ModuleNotFoundError:
    pass


__all__ = [
    'generate_random_portfolios',
    'backtest_strategy_parameters',
    'backtest_strategy_on_portfolios',
    'plot_heatmap',
    'calculate_positions',
    'calculate_trade_stats',
    'shuffle_ohlcv',
]


def generate_random_portfolios(universe: list[str], k: int, limit: int = None) -> list[tuple]:
    '''Generate randome portfolios with each containing k assets.

    Args:
        universe: A list of tickers
        k: The desired number of assets in portfolio
        limit: The maximum number of samples to generate. If None, generate all possible combinations. 

    Returns:
        A list of tuples, each being a portfolio of k assets.
    '''
    assert len(universe) > k
    if limit:
        samples = set()
        while len(samples) < limit:
            samples.add(tuple(sorted(random.sample(universe, k))))
        return list(samples)
    else:
        return list(itertools.combinations(universe, k))


def backtest_strategy_parameters(data: pd.DataFrame, strategy: Strategy,
                                 goals: list[str] = ['Return [%]', 'Max. Drawdown [%]', 'Calmar Ratio', 'SQN'],
                                 **kwargs: dict[str, Any]):
    '''Plot the strategy performance vs. a range of strategy parameters for visual inspection.

    Args:
        data: Price data used in `Backtest`
        strategy: A `Strategy` class under study
        goals: A list of goals as required by `Backtest.optimize()`. A plot is generated for each.
        **kwargs: Parameters to be passed to `Backtest.optimize()`, which should specify value range for strategy parameters.
    '''
    bt_args = {k: v for k, v in kwargs.items() if k in ['rebalance_tolerance', 'rebalance_cash_reserve', 'lot_size']}
    opt_args = {k: v for k, v in kwargs.items()
                if k not in ['rebalance_tolerance', 'rebalance_cash_reserve', 'lot_size']}
    bt = Backtest(data, strategy=strategy, fail_fast=False, **bt_args)
    for goal in goals:
        stats, heatmap = bt.optimize(maximize=goal, return_heatmap=True, **opt_args)
        if heatmap.index.nlevels == 1:
            heatmap.plot.bar(title=goal)
        else:
            heatmap = heatmap.unstack(0)
            _, ax = plt.subplots(figsize=(20, len(heatmap) * 0.5))
            ax.set_title(goal)
            sns.heatmap(heatmap, cmap='viridis')
        plt.show()
        display(stats.to_frame().transpose())
    return


def backtest_strategy_on_portfolios(
        data_lst: list[pd.DataFrame], strategy: Strategy, goal: str = 'Return [%]', **kwargs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Run strategy over a list of portfolios and collect their best performance after optimized strategy parameters.

    Args:
        data_lst: A list of price data used in `Backtest`, which implicitly define the portfolio and time range of backtesting.
        strategy: A `Strategy` class under study
        goal: Goal to be optimized for, as required by `Backtest.optimize()`
        **kwargs: Parameters to be passed to `Backtest.optimize()`, which should specify value range for strategy parameters.

    Returns:
        stats: best performance for each portfolio
        heatmap: a heatmap of portfolio performance vs. strategy parameters
    '''
    label_data = {','.join(data.columns.levels[0]): data for data in data_lst}
    _stats, _heatmap = {}, {}
    tq = tqdm(label_data.items(), leave=False)
    for label, data in tq:
        tq.set_description(str(label))
        bt = Backtest(data, strategy=strategy, fail_fast=False)
        s, h = bt.optimize(maximize=goal, return_heatmap=True, **kwargs)
        if s is not None and h is not None:
            _stats[label], _heatmap[label] = s, h
    stats, heatmap = pd.DataFrame(_stats).T, pd.DataFrame(_heatmap).T
    heatmap.attrs['goal'] = goal
    return stats, heatmap


def plot_heatmap(heatmap: pd.DataFrame, smooth: int | None = None):
    '''Plot a grid of heatmaps.

    Args:
        heatmap: A DataFrame as returned by `minitrade.backtest.utils.backtest_strategy_on_portfolios`
        smooth: If smooth is not None, use the lowest performance across a small window centered at a value as the performance for the parameter at the value
    '''
    # If smooth is given, use the lowest performance across a small window centered
    # at a value as the performance for the parameter at the value
    if smooth:
        heatmap = heatmap.rolling(smooth, center=True, axis=1).min()
    # sort portfolio by a rough sense of "overall" performance
    heatmap = heatmap.loc[heatmap.sum(axis=1).sort_values(ascending=False).index]
    # break plot into pages as too big a plot can have memory issue.
    for i in range(0, len(heatmap), 50):
        pg = heatmap.iloc[i:i+50]
        _, ax = plt.subplots(figsize=(pg.shape[1]//4+1, pg.shape[0]//4+1))
        ax.set_title(f'{heatmap.attrs.get("goal", "")} (rank {i} - {i + len(pg)})')
        ax.grid(False)
        sns.heatmap(pg, cmap='viridis')
        plt.savefig(f'performance_{i}-{i+len(pg)}.png', bbox_inches='tight')
        plt.show()


def calculate_positions(plan: 'TradePlan', orders: list[dict]):
    '''Calculate the final portfolio based on initial portfolio and order executed.'''
    tickers = plan.ticker_css.split(',')
    positions = {ticker: 0 for ticker in tickers}
    cash = plan.initial_cash
    if plan.initial_holding:
        for ticker, size in plan.initial_holding.items():
            positions[ticker] += size
    for order in orders:
        positions[order['ticker']] += order['size']
        cash -= order['size'] * order['entry_price'] + order['commission']
    return positions, cash


def calculate_trade_stats(data: pd.DataFrame, cash: int, orders: list[dict], holding: dict = {}):
    '''Create a day-by-day portolio report based on initial cash and order executed.

    Args:
        data: Price data used in `Backtest`
        cash: The initial cash to start with
        holding: The asset holding to start with
        orders: A list of orders executed. Each order is a dictionary like 
            {
                'ticker': ...,
                'entry_time': ...,
                'size': ...,
                'entry_price': ...,
                'commission': ...
            }
    '''

    def close_trade_helper(open_orders: list, order):
        '''Match order with previous orders on the opposite side to make completed trades'''
        closed_trades = []
        for open_order in open_orders[:]:
            if open_order['ticker'] == order['ticker']:
                if open_order['size'] * order['size'] < 0:
                    # if trade in the opposite direction, calculate how much of the original order can be closed
                    closed_size = copysign(min(abs(open_order['size']), abs(order['size'])), open_order['size'])
                    # create a trade record for the closed part
                    trade = {'ticker': order['ticker'],
                             'size': closed_size,
                             'entry_bar': open_order['entry_bar'],
                             'entry_time': open_order['entry_time'],
                             'entry_price': open_order['entry_price'],
                             'exit_bar': order['entry_bar'],
                             'exit_time': order['entry_time'],
                             'exit_price': order['entry_price'],
                             'pnl': (order['entry_price'] - open_order['entry_price']) * closed_size,
                             'return_pct[%]': ((order['entry_price'] / open_order['entry_price'] - 1) *
                                               copysign(1, closed_size) * 100) if open_order['entry_price'] else None}
                    closed_trades.append(trade)
                    # reduce the size of the original order
                    open_order['size'] -= closed_size
                    if open_order['size'] == 0:
                        open_orders.remove(open_order)
                    # update the size of the incoming order
                    order['size'] += closed_size
                    if order['size'] == 0:
                        break
                else:
                    # if position of the same side exists, break
                    break
        return closed_trades, order

    close_prices = data.xs('Close', axis=1, level=1)
    orders = orders.copy()
    orders.sort(key=lambda x: x['entry_time'])
    open_orders = []
    if holding:
        for ticker, size in holding.items():
            # Handle the preexisting assets as if they are bought at the first bar
            open_orders.append(
                {'ticker': ticker, 'entry_bar': 0, 'entry_time': data.index[0],
                 'size': size, 'entry_price': close_prices[ticker].iloc[0], 'commission': 0})
    trades = []
    if orders:
        order_df = pd.DataFrame(orders)
    else:
        order_df = pd.DataFrame(columns=['ticker', 'entry_time', 'size', 'entry_price', 'commission'])

    # match orders that close positions to those open positions in FIFO order to create completed trades
    while orders:
        order = orders.pop(0).copy()
        # each bar is denoted by the start time of the bar, therefore minus 1 to get the index
        order['entry_bar'] = (data.index < order['entry_time']).sum() - 1
        closed_trades, remaining_order = close_trade_helper(open_orders, order)
        trades.extend(closed_trades)
        if remaining_order['size'] != 0:
            open_orders.append(remaining_order)

    # create incompleted trades for remaining open positions
    alt_trades = trades.copy()
    for open_order in open_orders:
        trade = {'ticker': open_order['ticker'], 'size': open_order['size'], 'entry_bar': open_order['entry_bar'],
                 'entry_time': open_order['entry_time'], 'entry_price': open_order['entry_price'],
                 'exit_bar': None, 'exit_time': None, 'exit_price': None, 'pnl': None, 'return_pct[%]': None}
        trades.append(trade)
        # alternatively assuming all trades are closed on the last day using its close price
        exit_price = close_prices[open_order['ticker']].iloc[-1]
        alt_trade = {
            'ticker': open_order['ticker'],
            'size': open_order['size'],
            'entry_bar': open_order['entry_bar'],
            'entry_time': open_order['entry_time'],
            'entry_price': open_order['entry_price'],
            'exit_bar': len(data) - 1, 'exit_time': data.index[-1],
            'exit_price': exit_price, 'pnl': (exit_price - open_order['entry_price']) * open_order['size'],
            'return_pct[%]': ((exit_price / open_order['entry_price'] - 1) * copysign(1, open_order['size']) * 100)
            if open_order['entry_price'] else None, }
        alt_trades.append(alt_trade)

    # format into dataframe
    if trades:
        trade_df = pd.DataFrame(trades).sort_values('entry_time')
        alt_trade_df = pd.DataFrame(alt_trades)
    else:
        trade_df = pd.DataFrame(columns=['ticker', 'size', 'entry_bar', 'entry_time',
                                         'entry_price', 'exit_bar', 'exit_time', 'exit_price', 'pnl', 'return_pct[%]'])
        alt_trade_df = trade_df.copy()

    # calculate equity value over time
    equity = pd.Series(np.nan, index=data.index)
    for i in range(0, len(data)):
        if i < len(data) - 1:
            issued_orders = order_df[order_df['entry_time'] < data.index[i+1]]
        else:
            issued_orders = order_df
        cost = issued_orders['size'].dot(issued_orders['entry_price']) if len(issued_orders) > 0 else 0
        commission = issued_orders['commission'].sum()
        positions = issued_orders.groupby('ticker')['size'].sum()
        positions = pd.Series(positions, index=close_prices.iloc[i].index).fillna(0)
        if holding:
            positions = (positions + pd.Series(holding)).fillna(0)
        position_value = positions.dot(close_prices.iloc[i])
        equity.iloc[i] = cash + position_value - cost - commission
    cash_available = cash + order_df['size'].dot(order_df['entry_price']) - order_df['commission'].sum()

    # calculate ticker performance
    position_size = trade_df[trade_df['exit_time'].isna()].groupby('ticker')['size'].sum().rename('Position Size')
    realized_pnl = trade_df.groupby('ticker')['pnl'].sum().rename('Realized PnL')
    total_pnl = alt_trade_df.groupby('ticker')['pnl'].sum().rename('Total PnL')
    unrealized_pnl = (total_pnl - realized_pnl).rename('Unrealized PnL')
    pnl = pd.concat([position_size, unrealized_pnl, realized_pnl, total_pnl], axis=1)

    # calculate effective commission rate
    commission = order_df['commission'].sum()
    trade_value = order_df['size'].abs().dot(order_df['entry_price'])
    commission_rate = commission / trade_value if trade_value else 0

    return trade_df, equity, pnl, commission_rate, cash_available


def shuffle_ohlcv(data: pd.DataFrame, in_sync: bool = False, random_state: int = None):
    '''
    Generate shuffled OHLCV time-series with the same statistics as the original.
    '''
    def to_rr(df):
        rr = df[['Open', 'High', 'Low', 'Close']].div(df['Close'], axis=0) - 1
        rr['Close'] = df['Close'].pct_change().fillna(0)
        rr['Volume'] = df['Volume']
        return rr

    def to_ohlcv(df):
        df = df.copy()
        df['Close'] = (df['Close_rr'] + 1).cumprod() * df['Close'][0]
        df[['Open', 'High', 'Low']] = (df[['Open_rr', 'High_rr', 'Low_rr']]+1).mul(df['Close'], axis=0)
        df['Volume'] = df['Volume_rr']
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    rr = data.ta.apply(to_rr)
    if in_sync:
        rr.iloc[1:] = rr.iloc[1:].sample(frac=1, random_state=random_state, ignore_index=True)
    else:
        rr.iloc[1:] = rr.iloc[1:].ta.apply(lambda x: x.sample(frac=1, ignore_index=True, random_state=random_state))
    rr.index = data.index

    df = data.ta.join(rr, rsuffix='_rr').ta.apply(to_ohlcv)
    return df
