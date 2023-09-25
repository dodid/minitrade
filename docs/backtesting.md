---
hide:
  - toc
---

# Backtesting

`Minitrade` uses [Backtesting.py](https://github.com/kernc/backtesting.py) as the core library for backtesting and adds the capability to implement multi-asset strategies. 

## Single asset strategy

For single asset strategies, those written for Backtesting.py can be easily adapted to work with Minitrade. The following illustrates what changes are necessary:

```python
from minitrade.backtest import Strategy
from minitrade.backtest.core.lib import crossover

from minitrade.backtest.core.test import SMA


class SmaCross(Strategy):
    fast = 10
    slow = 20

    def init(self):
        price = self.data.Close.df
        self.ma1 = self.I(SMA, price, self.fast, overlay=True)
        self.ma2 = self.I(SMA, price, self.slow, overlay=True)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.position().close()
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.position().close()
            self.sell()


bt = Backtest(GOOG, SmaCross, commission=.001)
stats = bt.run()
bt.plot()
```

1. Change to import from minitrade modules. Generally `backtesting` becomes `minitrade.backtest.core`.
2. Minitrade expects `Volume` data to be always avaiable. `Strategy.data` should be consisted of OHLCV.
3. Minitrade doesn't try to guess where to plot the indicators. So if you want to overlay the indicators on the main chart, set `overlay=True` explicitly.
4. `Strategy.position` is no longer a property but a function. Any occurrence of `self.position` should be changed to `self.position()`. 

That's it. Check out [compatibility](compatibility.md) for more details.

![plot of single-asset strategy](https://imgur.com/N3E2d6m.jpg)

Also note that some original utility functions and strategy classes only make sense for single asset strategy. Don't use those in multi-asset strategies.

## Multi-asset strategy

`Minitrade` extends `Backtesting.py` to support backtesting of multi-asset strategies. 

Multi-asset strategies take a 2-level column DataFrame as data input. For example, for a strategy class that intends to invest in AAPL and GOOG as a portfolio, the `data` input to `Backtest()` should look like:

```python
# bt = Backtest(data, AaplGoogStrategy)
# print(data)

                          AAPL                              GOOG 
                          Open  High  Low   Close Volume    Open  High  Low   Close Volume
Date          
2018-01-02 00:00:00-05:00 40.39 40.90 40.18 40.89 102223600 52.42 53.35 52.26 53.25 24752000
2018-01-03 00:00:00-05:00 40.95 41.43 40.82 40.88 118071600 53.22 54.31 53.16 54.12 28604000
2018-01-04 00:00:00-05:00 40.95 41.18 40.85 41.07 89738400 54.40 54.68 54.20 54.32 20092000
2018-01-05 00:00:00-05:00 41.17 41.63 41.08 41.54 94640000 54.70 55.21 54.60 55.11 25582000
2018-01-08 00:00:00-05:00 41.38 41.68 41.28 41.38 82271200 55.11 55.56 55.08 55.35 20952000
```

Like in `Backtesting.py`, `self.data`, when accessed from within `Strategy.init()`, is `_Data` type that supports progressively revealing of data, and the raw DataFrame can be accessed by `self.data.df`. 

```python
# When called from within Strategy.init()

# self.data
<Data i=4 (2018-01-08 00:00:00-05:00) ('AAPL', 'Open')=41.38, ('AAPL', 'High')=41.68, ('AAPL', 'Low')=41.28, ('AAPL', 'Close')=41.38, ('AAPL', 'Volume')=82271200.0, ('GOOG', 'Open')=55.11, ('GOOG', 'High')=55.56, ('GOOG', 'Low')=55.08, ('GOOG', 'Close')=55.35, ('GOOG', 'Volume')=20952000.0>

# self.data.df
                          AAPL                              GOOG 
                          Open  High  Low   Close Volume    Open  High  Low   Close Volume
dt          
2018-01-02 00:00:00-05:00 40.39 40.90 40.18 40.89 102223600 52.42 53.35 52.26 53.25 24752000
2018-01-03 00:00:00-05:00 40.95 41.43 40.82 40.88 118071600 53.22 54.31 53.16 54.12 28604000
2018-01-04 00:00:00-05:00 40.95 41.18 40.85 41.07 89738400 54.40 54.68 54.20 54.32 20092000
2018-01-05 00:00:00-05:00 41.17 41.63 41.08 41.54 94640000 54.70 55.21 54.60 55.11 25582000
2018-01-08 00:00:00-05:00 41.38 41.68 41.28 41.38 82271200 55.11 55.56 55.08 55.35 20952000
```

To facilitate indicator calculation, Minitrade has built-in integration with [pandas_ta](https://github.com/twopirllc/pandas-ta) as TA library. `pandas_ta` is accessible using `.ta` property of any DataFrame. Check out [here](https://github.com/twopirllc/pandas-ta#pandas-ta-dataframe-extension) for usage. `.ta` is also enhanced to support 2-level DataFrames. 

For example,

```python
# print(self.data.df.ta.sma(3))

                                 AAPL       GOOG
Date                                            
2018-01-02 00:00:00-05:00         NaN        NaN
2018-01-03 00:00:00-05:00         NaN        NaN
2018-01-04 00:00:00-05:00   40.946616  53.898000
2018-01-05 00:00:00-05:00   41.163408  54.518500
2018-01-08 00:00:00-05:00   41.331144  54.926167
```

Even simpler, `self.data.ta.sma(3)` works the same on `self.data`.


`self.I()` can take both DataFrame/Series and functions as arguments to define an indicator. If DataFrame/Series is given as input, it's expected to have exactly the same index as `self.data`. For example,

```
self.sma = self.I(self.data.df.ta.sma(3), name='SMA_3')
```

Within `Strategy.next()`, indicators are returned as type `_Array`, essentially `numpy.ndarray`, same as in `Backtesting.py`. The `.df` accessor returns either `DataFrame` or `Series` of the corresponding value. It's the caller's responsibility to know which exact type should be returned. `.s` accessor is also available but only as a syntax suger to return a `Series`. If the actual data is a DataFrame, `.s` throws a `ValueError`. 

A key addition to support multi-asset strategy is a `Strategy.alloc` attribute, which combined with `Strategy.rebalance()`, allows to specify how cash value should be allocate among the different assets. 

Here is an example:

```python
# This strategy evenly allocates cash into the assets
# that have the top 2 highest rate-of-change every day, 
# on condition that the ROC is possitive.

class TopPositiveRoc(Strategy):

    n = 10

    def init(self):
        roc = self.data.ta.roc(self.n)
        self.roc = self.I(roc, name='ROC')

    def next(self):
        roc = self.roc.df.iloc[-1]
        self.alloc.add(roc.nlargest(2).index, roc > 0).equal_weight()
        self.rebalance()
```

`self.alloc` keeps track of what assets to be bought and how much weight in term of portfolio value is allocated to each. 

At the beginning of each `Strategy.next()` call, `self.alloc` starts empty. 

Use `alloc.add()` to add assets to a candidate pool. `alloc.add()` takes either a list-like structure or a boolean Series as input. If it's a list-like structure, all assets in the list are added to the pool. If it's a boolean Series, index items having a `True` value are added to the pool. When multiple conditions are specified in the same call, the conditions are joined by logical `AND` and the resulted assets are added the the pool. `alloc.add()` can be called multiple times which means a logical `OR` relation and add all assets involved to the pool. 

Once candidate assets are determined, Call `alloc.equal_weight()` to assign equal weight in term of value to each selected asset.

And finally, call `Strategy.rebalance()`, which will look at the current equity value, calculate the target value for each asset, calculate how many shares to buy or sell based on the current long/short positions, and generate orders that will bring the portfolio to the target allocation.

Run the above strategy on some DJIA components: 

![plot of multi-asset strategy](https://imgur.com/ecy6yTm.jpg)


## Running backtest

Once a strategy is defined, you can test it as follows:

```python
from minitrade.datasource import QuoteSource
from minitrade.backtest import Backtest

yahoo = QuoteSource.get_source('Yahoo')
data = yahoo.daily_bar('AAPL', start='2018-01-01', end='2019-01-01')
bt = Backtest(data, SmaCross)
stats = bt.run()
```

First instantiate a built-in data source that gets quotes from Yahoo Finance. Then acquire daily bars for the specific stock symbol and date range. This returns a Dataframe in expected 2-level column format. Note currently daily bar is the only supported data frequency. 

```python

                            AAPL                                
                            Open   High    Low  Close     Volume
dt                                                              
2018-01-02 00:00:00-05:00  40.28  40.79  40.07  40.78  102223600
2018-01-03 00:00:00-05:00  40.84  41.32  40.71  40.77  118071600
2018-01-04 00:00:00-05:00  40.84  41.06  40.73  40.96   89738400
2018-01-05 00:00:00-05:00  41.06  41.51  40.96  41.43   94640000
2018-01-08 00:00:00-05:00  41.27  41.57  41.17  41.27   82271200
...                          ...    ...    ...    ...        ...
2018-12-24 00:00:00-05:00  35.60  36.41  35.22  35.28  148676800
2018-12-26 00:00:00-05:00  35.63  37.78  35.25  37.76  234330000
2018-12-27 00:00:00-05:00  37.44  37.67  36.06  37.52  212468400
2018-12-28 00:00:00-05:00  37.84  38.09  37.13  37.54  169165600
2018-12-31 00:00:00-05:00  38.09  38.29  37.60  37.90  140014000
```

Next, create a backtest instance by supplying the data and strategy at a minimum. You don't need to specify the date range for backtesting, which is inferred from the data. `Backtest()` allows you to specify some other interesting parameters, such as `cash` for initial cash investment, `commission` for commission rate, `trade_on_close` for if trade should happen on market open or market close, etc. See [API Reference](backtest.md) for usage. 

Finally, call `bt.run()` to run the backtest.

You can examine the strategy performance as stored in `stats`:

```python
# print(stats)
Start                                             2018-01-02 00:00:00-05:00
End                                               2018-12-31 00:00:00-05:00
Duration                                                  363 days 00:00:00
Exposure Time [%]                                                 85.657371
Equity Final [$]                                                9785.368222
Equity Peak [$]                                                10463.023909
Return [%]                                                        -2.146318
Buy & Hold Return [%]                                             -7.054367
Return (Ann.) [%]                                                 -2.154776
Volatility (Ann.) [%]                                             23.856348
Sharpe Ratio                                                      -0.090323
Sortino Ratio                                                     -0.132079
Calmar Ratio                                                      -0.071304
Max. Drawdown [%]                                                -30.219472
Avg. Drawdown [%]                                                -16.217719
Max. Drawdown Duration                                    295 days 00:00:00
Avg. Drawdown Duration                                    153 days 00:00:00
# Trades                                                                 10
Win Rate [%]                                                           30.0
Best Trade [%]                                                    27.343079
Worst Trade [%]                                                  -13.749999
Avg. Trade [%]                                                    -0.223291
Max. Trade Duration                                        75 days 00:00:00
Avg. Trade Duration                                        32 days 00:00:00
Profit Factor                                                      1.094006
Expectancy [%]                                                     0.370883
SQN                                                               -0.071592
Kelly Criterion                                                   -0.019763
_strategy                                                          SmaCross
_equity_curve                                              Equity       ...
_trades                      EntryBar  ExitBar Ticker  Size  EntryPrice ...
_orders                                             Ticker  Side  Size
S...
_positions                                   {'AAPL': -147, 'Margin': 4214}
dtype: object
```

You can also visually inspect the backtest progress by

```python
bt.plot()
```

![plot of minitrade backtest](https://imgur.com/rBnfSLu.png)

## Parameter optimization

To search for the optimial parameters for a strategy, you can run the following:

```python
stats, heatmap = bt.optimize(
    fast=range(10, 110, 10),
    slow=range(20, 210, 20),
    constraint=lambda p: p.fast < p.slow,
    maximize='Equity Final [$]',
    random_state=0,
    return_heatmap=True)
```

```python
# print(heatmap)
fast  slow
10    20       9785.385258
      40      12563.217058
      60      11407.502889
      80      13688.424394
      100     11766.968888
20    40      11951.293618
      60      11701.255333
      80      13123.190833
      100     11715.922927
30    40      10410.359668
      60      10309.179288
      80      11793.435469
      100     11220.492771
40    60      14623.910118
      80      11154.921753
      100     11072.236436
50    60      12682.605254
      80      10979.434926
      100     11444.498170
Name: Equity Final [$], dtype: float64
```

Plot the result as a heatmap:

```python
import seaborn as sns

sns.heatmap(heatmap.groupby(['slow', 'fast']).mean().unstack(), cmap='viridis')
```

![Minitrade optimize heatmap](https://imgur.com/fetS4MU.png)

