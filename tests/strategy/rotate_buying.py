import numpy as np
import pandas as pd

from minitrade.backtest import Strategy


class RotateBuying(Strategy):
    '''A dumb strategy that buy a different asset everyday'''

    def init(self):
        self.weight = pd.Series(range(len(self.data.tickers)), index=self.data.tickers) == 0

    def next(self):
        self.weight = pd.Series(np.roll(self.weight, 1), index=self.data.tickers)
        self.alloc.add(self.weight).equal_weight()
        self.rebalance()
