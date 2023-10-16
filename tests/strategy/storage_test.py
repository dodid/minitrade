import numpy as np
import pandas as pd

from minitrade.backtest import Strategy


class StorageTestStrategy(Strategy):
    '''A dumb strategy that manipulates the storage'''

    def init(self):
        print('storage: ', self.storage)
        if not self.storage:
            self.storage['int'] = 0
            self.storage['float'] = 0.0
            self.storage['str'] = '0'
            self.storage['list'] = [0]
            self.storage['dict'] = {'int': 0}
            self.storage['np'] = np.array([0])
            self.storage['pd'] = pd.DataFrame({'int': [0]})
            self.firstrun = True
        else:
            self.firstrun = False

    def next(self):
        if self.storage and not self.firstrun:
            self.storage['int'] = 1
            self.storage['float'] = 1.0
            self.storage['str'] = '1'
            self.storage['list'] = [1]
            self.storage['dict'] = {'int': 1}
            self.storage['np'] = np.array([1])
            self.storage['pd'] = pd.DataFrame({'int': [1]})
