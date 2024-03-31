from minitrade.backtest import Strategy


class RotateBuying(Strategy):
    '''A dumb strategy that buy a different asset everyday'''

    def init(self):
        pass

    def next(self):
        self.alloc.assume_zero()
        index = len(self.data) % len(self.alloc.tickers)
        self.alloc.weights.iloc[index] = 1
        self.rebalance(cash_reserve=0.5)
