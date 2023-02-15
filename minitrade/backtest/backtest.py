
from math import copysign

import numpy as np
import pandas as pd


def portfolio_report(data: pd.DataFrame, init_cash: int, orders: list[dict]):
    '''
    Create day by day portolio report based on initial cash and order executed.
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
                             'return_pct[%]': (order['entry_price'] / open_order['entry_price'] - 1) *
                             copysign(1, closed_size) * 100}
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
            'exit_price': exit_price,
            'pnl': (exit_price - open_order['entry_price']) * open_order['size'],
            'return_pct[%]': (exit_price / open_order['entry_price'] - 1) * copysign(1, open_order['size']) * 100}
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
        pnl = -issued_orders['size'].dot(issued_orders['entry_price']) if len(issued_orders) > 0 else 0
        commission = issued_orders['commission'].sum()
        positions = issued_orders.groupby('ticker')['size'].sum()
        positions = pd.Series(positions, index=close_prices.iloc[i].index).fillna(0)  # align the index
        position_value = positions.dot(close_prices.iloc[i])
        equity.iloc[i] = init_cash + pnl - commission + position_value

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

    return order_df, trade_df, equity, pnl, commission_rate
