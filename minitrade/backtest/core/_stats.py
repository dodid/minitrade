import warnings
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd

from ._util import _data_period

if TYPE_CHECKING:
    from .backtesting import Order, Strategy, Trade


def compute_drawdown_duration_peaks(dd: pd.Series):
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())
    df = df[df['iloc'] > df['prev'] + 1].astype(int)

    # If no drawdown since no trade, avoid below for pandas sake and return nan series
    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
    df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
    df = df.reindex(dd.index)
    return df['duration'], df['peak_dd']


def geometric_mean(returns: pd.Series) -> float:
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1


def compute_stats(
        orders: Union[List['Order'], pd.DataFrame],
        trades: Union[List['Trade'], pd.DataFrame],
        equity: pd.DataFrame,
        ohlc_data: pd.DataFrame,
        strategy_instance: 'Strategy',
        risk_free_rate: float = 0,
        positions: dict = None,
        trade_start_bar: int = 0,
) -> pd.Series:
    assert -1 < risk_free_rate < 1

    index = ohlc_data.index
    dd = 1 - equity['Equity'] / np.maximum.accumulate(equity['Equity'])
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))

    if isinstance(orders, pd.DataFrame):
        orders_df = orders
    else:
        orders_df = pd.DataFrame({
            'SignalTime': [t.entry_time for t in orders],
            'Ticker': [t.ticker for t in orders],
            'Side': ['Buy' if t.size > 0 else 'Sell' for t in orders],
            'Size': [int(t.size) for t in orders],
        }).set_index('SignalTime')

    equity_df = pd.concat([equity, pd.DataFrame({'DrawdownPct': dd, 'DrawdownDuration': dd_dur}, index=index)], axis=1)

    if isinstance(trades, pd.DataFrame):
        trades_df = trades
    else:
        # Came straight from Backtest.run()
        trades_df = pd.DataFrame({
            'EntryBar': [t.entry_bar for t in trades],
            'ExitBar': [t.exit_bar for t in trades],
            'Ticker': [t.ticker for t in trades],
            'Size': [t.size for t in trades],
            'EntryPrice': [t.entry_price for t in trades],
            'ExitPrice': [t.exit_price for t in trades],
            'PnL': [t.pl for t in trades],
            'ReturnPct': [t.pl_pct for t in trades],
            'EntryTime': [t.entry_time for t in trades],
            'ExitTime': [t.exit_time for t in trades],
            'Tag': [t.tag for t in trades],
        })
        trades_df['Duration'] = trades_df['ExitTime'] - trades_df['EntryTime']
    del trades

    pl = trades_df['PnL']
    returns = trades_df['ReturnPct']
    durations = trades_df['Duration']

    def _round_timedelta(value, _period=_data_period(index)):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(_period, 'resolution_string', None) or _period.resolution
        return value.ceil(resolution)

    s = pd.Series(dtype=object)
    s.loc['Start'] = index[0]
    s.loc['End'] = index[-1]
    s.loc['Duration'] = s.End - s.Start

    have_position = np.repeat(0, len(index))
    for t in trades_df.itertuples(index=False):
        have_position[t.EntryBar:t.ExitBar + 1] = 1

    s.loc['Exposure Time [%]'] = have_position.mean() * 100  # In "n bars" time, not index time
    s.loc['Equity Final [$]'] = equity['Equity'].iloc[-1]
    s.loc['Equity Peak [$]'] = equity['Equity'].max()
    s.loc['Return [%]'] = (equity['Equity'].iloc[-1] - equity['Equity'].iloc[0]) / equity['Equity'].iloc[0] * 100
    c = ohlc_data.Close.values
    s.loc['Buy & Hold Return [%]'] = (c[-1] - c[trade_start_bar]) / c[trade_start_bar] * 100  # long-only return

    gmean_period_return: float = 0
    period_returns = np.array(np.nan)
    annual_trading_periods = np.nan
    if isinstance(index, pd.DatetimeIndex):
        period = equity.index.to_series().diff().mean().days
        if period <= 1:
            period_returns = equity_df['Equity'].resample('D').last().dropna().pct_change()
            gmean_period_return = geometric_mean(period_returns)
            annual_trading_periods = float(
                365 if index.dayofweek.to_series().between(5, 6).mean() > 2/7 * .6 else 252)
        elif period >= 28 and period <= 31:
            period_returns = equity_df['Equity'].pct_change()
            gmean_period_return = geometric_mean(period_returns)
            annual_trading_periods = 12
        elif period >= 365 and period <= 366:
            period_returns = equity_df['Equity'].pct_change()
            gmean_period_return = geometric_mean(period_returns)
            annual_trading_periods = 1
        else:
            warnings.warn(f'Unsupported data period from index: {period} days.')

    # Annualized return and risk metrics are computed based on the (mostly correct)
    # assumption that the returns are compounded. See: https://dx.doi.org/10.2139/ssrn.3054517
    # Our annualized return matches `empyrical.annual_return(day_returns)` whereas
    # our risk doesn't; they use the simpler approach below.
    annualized_return = (1 + gmean_period_return)**annual_trading_periods - 1
    s.loc['Return (Ann.) [%]'] = annualized_return * 100
    s.loc['Volatility (Ann.) [%]'] = np.sqrt((period_returns.var(ddof=int(bool(period_returns.shape))) + (1 + gmean_period_return)**2)**annual_trading_periods - (1 + gmean_period_return)**(2*annual_trading_periods)) * 100  # noqa: E501
    # s.loc['Return (Ann.) [%]'] = gmean_day_return * annual_trading_days * 100
    # s.loc['Risk (Ann.) [%]'] = day_returns.std(ddof=1) * np.sqrt(annual_trading_days) * 100

    # Our Sharpe mismatches `empyrical.sharpe_ratio()` because they use arithmetic mean return
    # and simple standard deviation
    s.loc['Sharpe Ratio'] = (s.loc['Return (Ann.) [%]'] - risk_free_rate) / (s.loc['Volatility (Ann.) [%]'] or np.nan)  # noqa: E501
    with warnings.catch_warnings():
        # wrap to catch RuntimeWarning: divide by zero encountered in scalar divide
        warnings.filterwarnings('error')
        try:
            # Our Sortino mismatches `empyrical.sortino_ratio()` because they use arithmetic mean return
            s.loc['Sortino Ratio'] = (annualized_return - risk_free_rate) / (np.sqrt(np.mean(period_returns.clip(-np.inf, 0)**2)) * np.sqrt(annual_trading_periods))  # noqa: E501
        except Warning:
            s.loc['Sortino Ratio'] = np.nan
    max_dd = -np.nan_to_num(dd.max())
    s.loc['Calmar Ratio'] = annualized_return / (-max_dd or np.nan)
    s.loc['Max. Drawdown [%]'] = max_dd * 100
    s.loc['Avg. Drawdown [%]'] = -dd_peaks.mean() * 100
    s.loc['Max. Drawdown Duration'] = _round_timedelta(dd_dur.max())
    s.loc['Avg. Drawdown Duration'] = _round_timedelta(dd_dur.mean())
    s.loc['# Trades'] = n_trades = len(trades_df)
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    s.loc['Win Rate [%]'] = win_rate * 100
    s.loc['Best Trade [%]'] = returns.max() * 100
    s.loc['Worst Trade [%]'] = returns.min() * 100
    mean_return = geometric_mean(returns)
    s.loc['Avg. Trade [%]'] = mean_return * 100
    s.loc['Max. Trade Duration'] = _round_timedelta(durations.max())
    s.loc['Avg. Trade Duration'] = _round_timedelta(durations.mean())
    s.loc['Profit Factor'] = returns[returns > 0].sum() / (abs(returns[returns < 0].sum()) or np.nan)  # noqa: E501
    s.loc['Expectancy [%]'] = returns.mean() * 100
    s.loc['SQN'] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)
    s.loc['Kelly Criterion'] = win_rate - (1 - win_rate) / (pl[pl > 0].mean() / -pl[pl < 0].mean())

    s.loc['_strategy'] = strategy_instance
    s.loc['_equity_curve'] = equity_df
    s.loc['_trades'] = trades_df
    s.loc['_orders'] = orders_df
    s.loc['_positions'] = positions
    s.loc['_trade_start_bar'] = trade_start_bar

    s = _Stats(s)
    return s


class _Stats(pd.Series):
    def __repr__(self):
        # Prevent expansion due to Equity and _trades dfs
        with pd.option_context('max_colwidth', 20):
            return super().__repr__()
