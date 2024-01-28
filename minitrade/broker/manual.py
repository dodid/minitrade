from __future__ import annotations

import typing
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from minitrade.broker import Broker, BrokerAccount
from minitrade.broker.base import OrderValidator
from minitrade.trader import RawOrder, TradePlan
from minitrade.utils.mtdb import MTDB

if typing.TYPE_CHECKING:
    from minitrade.trader import RawOrder, TradePlan


class ManualBroker(Broker):
    '''
    ManualBroker is a mock broker that assumes backtest orders will be placed manually.
    '''

    def __init__(self, account: BrokerAccount):
        self.account = account

    def is_ready(self) -> bool:
        return True

    def connect(self):
        pass

    def submit_order(self, plan: TradePlan, order: RawOrder) -> str:
        validator = OrderValidator(plan)
        validator.validate(order)
        order_id = MTDB.uniqueid()
        order = asdict(order)
        order['broker_order_id'] = order_id
        order['submit_time'] = datetime.utcnow()
        MTDB.save('ManualTrade', order)
        return order_id

    def cancel_order(self, plan: TradePlan, order: RawOrder = None) -> bool:
        pass

    def get_account_info(self) -> dict:
        return {'about': 'A mock broker that assumes you will submit orders manually'}

    def get_portfolio(self) -> pd.DataFrame | None:
        trades = MTDB.get_all('ManualTrade', cls=dict)
        if trades:
            df = pd.DataFrame(trades)
            df = df[df['trade_time'].notnull()]     # only count filled trades
            return df.groupby('ticker')['size'].sum() if len(df) > 0 else None
        else:
            return None

    def download_trades(self) -> pd.DataFrame | None:
        trades = MTDB.get_all('ManualTrade', orderby=('submit_time', False), cls=dict)
        return pd.DataFrame(trades) if trades else None

    def download_orders(self) -> pd.DataFrame | None:
        return None

    def disconnect(self) -> None:
        pass

    def resolve_tickers(self, ticker_css: str) -> dict[str, list]:
        return {ticker: [{'id': ticker, 'label': ticker}] for ticker in ticker_css.split(',')}

    def find_trades(self, order: RawOrder) -> dict:
        return MTDB.get_all('ManualTrade', 'id', order.id, cls=dict)

    def find_order(self, order: RawOrder) -> dict:
        return MTDB.get_one('ManualTrade', 'id', order.id, cls=dict)

    def format_trades(self, orders: list[RawOrder]) -> list[dict]:
        trades = []
        for order in orders:
            for status in self.find_trades(order):
                if status['trade_time'] and status['price']:
                    trade = {'ticker': order.ticker,
                             'entry_time': status['trade_time'],
                             'size': status['size'],
                             'entry_price': float(status['price']),
                             'commission': float(status['commission'])}
                    trades.append(trade)
        return trades

    def update_trade(self, id, price, commission, trade_time):
        MTDB.update('ManualTrade', 'id', id, values={'price': price,
                    'commission': commission, 'trade_time': trade_time})
