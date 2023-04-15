---
hide:
  - toc
---

# Writing strategy

Suppose we want to develope `SomeStrategy` that invests in AAPL and GOOG. 

This is how we will run the backtest:

```python
yahoo = QuoteSource.get_source('Yahoo')
data = yahoo.daily_bar('AAPL,GOOG', start='2018-01-01', end='2018-01-08')
bt = Backtest(data, SomeStrategy)
bt.run()
```

And `SomeStrategy` looks like the following:

```python
class SomeStrategy(Strategy):
    
    def init(self):
        # prepare indicators

    def next(self):
        # process data bar and generate orders
```

### `Strategy.init()`

#### Accessing `self.data`

In `Strategy.init()`, we calculate indicators from asset price data, which can be accessed either by `self.data` or by `self.data.df`. The former is closer to Numpy array, while the later is a Pandas DataFrame. 

The following illustrates how to slice the data in different, sometimes equivalent, ways:

```python
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

# self.data.Close
[[40.89 53.25]
 [40.88 54.12]
 [41.07 54.32]
 [41.54 55.11]
 [41.38 55.35]]

# self.data.Close.df
                            AAPL   GOOG
dt                                     
2018-01-02 00:00:00-05:00  40.89  53.25
2018-01-03 00:00:00-05:00  40.88  54.12
2018-01-04 00:00:00-05:00  41.07  54.32
2018-01-05 00:00:00-05:00  41.54  55.11
2018-01-08 00:00:00-05:00  41.38  55.35

# self.data.df.xs("Close", axis=1, level=1)
                            AAPL   GOOG
dt                                     
2018-01-02 00:00:00-05:00  40.89  53.25
2018-01-03 00:00:00-05:00  40.88  54.12
2018-01-04 00:00:00-05:00  41.07  54.32
2018-01-05 00:00:00-05:00  41.54  55.11
2018-01-08 00:00:00-05:00  41.38  55.35

# self.data.Close[-1]
[41.38 55.35]

# self.data.Close.df.iloc[-1]
AAPL    41.38
GOOG    55.35
Name: 2018-01-08 00:00:00-05:00, dtype: float64

# self.data["AAPL"]
[[4.039000e+01 4.090000e+01 4.018000e+01 4.089000e+01 1.022236e+08]
 [4.095000e+01 4.143000e+01 4.082000e+01 4.088000e+01 1.180716e+08]
 [4.095000e+01 4.118000e+01 4.085000e+01 4.107000e+01 8.973840e+07]
 [4.117000e+01 4.163000e+01 4.108000e+01 4.154000e+01 9.464000e+07]
 [4.138000e+01 4.168000e+01 4.128000e+01 4.138000e+01 8.227120e+07]]

# self.data["AAPL"].df
                            Open   High    Low  Close     Volume
dt                                                              
2018-01-02 00:00:00-05:00  40.39  40.90  40.18  40.89  102223600
2018-01-03 00:00:00-05:00  40.95  41.43  40.82  40.88  118071600
2018-01-04 00:00:00-05:00  40.95  41.18  40.85  41.07   89738400
2018-01-05 00:00:00-05:00  41.17  41.63  41.08  41.54   94640000
2018-01-08 00:00:00-05:00  41.38  41.68  41.28  41.38   82271200

# self.data.df["AAPL"]
                            Open   High    Low  Close     Volume
dt                                                              
2018-01-02 00:00:00-05:00  40.39  40.90  40.18  40.89  102223600
2018-01-03 00:00:00-05:00  40.95  41.43  40.82  40.88  118071600
2018-01-04 00:00:00-05:00  40.95  41.18  40.85  41.07   89738400
2018-01-05 00:00:00-05:00  41.17  41.63  41.08  41.54   94640000
2018-01-08 00:00:00-05:00  41.38  41.68  41.28  41.38   82271200

# self.data["AAPL", "Close"]
[40.89 40.88 41.07 41.54 41.38]

# self.data["AAPL", "Close"].df
dt
2018-01-02 00:00:00-05:00    40.89
2018-01-03 00:00:00-05:00    40.88
2018-01-04 00:00:00-05:00    41.07
2018-01-05 00:00:00-05:00    41.54
2018-01-08 00:00:00-05:00    41.38
Name: (AAPL, Close), dtype: float64

# self.data.df[("AAPL","Close")]
dt
2018-01-02 00:00:00-05:00    40.89
2018-01-03 00:00:00-05:00    40.88
2018-01-04 00:00:00-05:00    41.07
2018-01-05 00:00:00-05:00    41.54
2018-01-08 00:00:00-05:00    41.38
Name: (AAPL, Close), dtype: float64

# self.data["AAPL", "Close"][-1]
41.38

# self.data["AAPL", "Close"].df[-1]
41.38

# self.data.df[("AAPL","Close")][-1]
41.38
```

Since `Strategy.init()` is only called once, performance is generally not an issue. It's recommended to use accessors that return DataFrame to simplify further processing.

Here are some other properties of `self.data`:

```python
# self.data.tickers
['AAPL', 'GOOG']

# self.data.index
DatetimeIndex(['2018-01-02 00:00:00-05:00', '2018-01-03 00:00:00-05:00',
               '2018-01-04 00:00:00-05:00', '2018-01-05 00:00:00-05:00',
               '2018-01-08 00:00:00-05:00'],
              dtype='datetime64[ns, America/New_York]', name='dt', freq=None)
```

#### Creating indicators

Minitrade has built-in integration with [pandas_ta](https://github.com/twopirllc/pandas-ta) as TA library. `pandas_ta` provides a number of technical indicators via a `.ta` extension added to every DataFrame. And Minitrade extends `.ta` to support 2-level column DataFrame. 

For example:

```python
# self.data.df.ta.log_return(cumulative=True) 
# Or simply, self.data.ta.log_return(cumulative=True)
                               AAPL      GOOG
dt                                           
2018-01-02 00:00:00-05:00  0.000000  0.000000
2018-01-03 00:00:00-05:00 -0.000245  0.016206
2018-01-04 00:00:00-05:00  0.004392  0.019895
2018-01-05 00:00:00-05:00  0.015771  0.034333
2018-01-08 00:00:00-05:00  0.011912  0.038679

# self.data.df.ta.sma(3)
# Or simply, self.data.ta.sma(3)
                                AAPL       GOOG
dt                                             
2018-01-02 00:00:00-05:00        NaN        NaN
2018-01-03 00:00:00-05:00        NaN        NaN
2018-01-04 00:00:00-05:00  40.946667  53.896667
2018-01-05 00:00:00-05:00  41.163333  54.516667
2018-01-08 00:00:00-05:00  41.330000  54.926667

# self.data["AAPL"].df.ta.sma(3)
dt
2018-01-02 00:00:00-05:00          NaN
2018-01-03 00:00:00-05:00          NaN
2018-01-04 00:00:00-05:00    40.946667
2018-01-05 00:00:00-05:00    41.163333
2018-01-08 00:00:00-05:00    41.330000
Name: SMA_3, dtype: float64
```

Use [Strategy.I()](https://dodid.github.io/minitrade/backtest/#minitrade.backtest.core.backtesting.Strategy.I) to register indicators to make them available in `Strategy.next()`. It accepts any of the following:

- A DataFrame or Series that has the same index as `self.data.index`.
- A list or Numpy array which will be converted into DataFrame or Series by using `self.data.index` as index. Therefore, its length must be equal to `len(self.data.index)`. 
- A function that returns one of the above.

For example,
```python
self.sma = self.I(self.data.ta.sma(3), name='SMA_3')

# self.sma
                                AAPL       GOOG
dt                                             
2018-01-02 00:00:00-05:00        NaN        NaN
2018-01-03 00:00:00-05:00        NaN        NaN
2018-01-04 00:00:00-05:00  40.946667  53.896667
2018-01-05 00:00:00-05:00  41.163333  54.516667
2018-01-08 00:00:00-05:00  41.330000  54.926667
```

Note `self.sma` is a DataFrame if accessed in `Strategy.init()`. And we will see it's changed to Numpy array if accessed in `Strategy.next()`.

### Strategy.next()

#### Accessing data

In `Strategy.next()`, both `self.data` and registered indicators are revealed progressively to prevent look-ahead bias. 

Since `Strategy.next()` is called for every bar in a backtest and optimizing a strategy may take many backtest runs, data indexing performance can be a concern in `Strategy.next()`. Therefore, indicators are returned as Numpy array by default, for example:

```python
# self.sma at time step 4
[[        nan         nan]
 [        nan         nan]
 [40.94666667 53.89666667]
 [41.16333333 54.51666667]]
```

To access the DataFrame version of indicators, use `.df` property:

```python
# self.sma.df at time step 4
                                AAPL       GOOG
dt                                             
2018-01-02 00:00:00-05:00        NaN        NaN
2018-01-03 00:00:00-05:00        NaN        NaN
2018-01-04 00:00:00-05:00  40.946667  53.896667
2018-01-05 00:00:00-05:00  41.163333  54.516667
```

To get the current time step, use `len(self.data)`. 

To get the current simulation time, use `self.data.now`.

#### Rebalancing

A key addition to support multi-asset strategy is a `Strategy.alloc` attribute, which combined with `Strategy.rebalance()` API, allows to specify how cash value should be allocate among the different assets. 

Here is an example:

```python
# This strategy evenly allocates cash into the assets
# that have the top 2 highest rate-of-change every day, 
# on condition that the ROC is possitive.

class TopPositiveRoc(Strategy):
    n = 10

    def init(self):
        self.roc = self.I(self.data.ta.roc(self.n), name='ROC')

    def next(self):
        roc = self.roc.df.iloc[-1]
        self.alloc.add(
            roc.nlargest(2).index, 
            roc > 0
        ).equal_weight()
        self.rebalance()
```

`self.alloc` keeps track of what assets to be bought and how much weight is assigned to each. 

At the beginning of each `Strategy.next()` call, `self.alloc` starts empty. 

Use `alloc.add()` to add assets to a candidate pool. `alloc.add()` takes either a list-like structure or a boolean Series as input. If it's a list-like structure, all assets in the list are added to the pool. If it's a boolean Series, index items having a `True` value are added to the pool. When multiple conditions are specified in the same call, the conditions are joined by logical `AND` and the resulted assets are added the the pool. `alloc.add()` can be called multiple times which means a logical `OR` relation and add all assets involved to the pool. 

`alloc.drop()` works similarly but removes assets from the pool.

Once candidate assets are determined, Call `alloc.equal_weight()` to assign equal weight in term of value to each selected asset.

`self.alloc` can also be assigned to directly to give full control of asset allocation. For example, to split equally among all assets:

```python
self.alloc.current = pd.Series([1 / len(self.data.tickers)] * len(self.data.tickers), index=self.data.tickers)

# self.alloc.current
AAPL    0.5
GOOG    0.5
Name: c, dtype: float64

self.rebalance()
```

And finally, call `Strategy.rebalance()`, which will look at the current equity value, calculate the target value for each asset, calculate how many shares to buy or sell based on the current long/short positions, and generate orders that will bring the portfolio to the target allocation.

### Single asset strategy

For single asset strategy, the following illustrates how `self.data` can be accessed when there is only one asset.

```
yahoo = QuoteSource.get_source('Yahoo')
data = yahoo.daily_bar('AAPL', start='2018-01-01', end='2018-01-08')
bt = Backtest(data, SomeStrategy)
```

```
# self.data.the_ticker
AAPL

# self.data
<Data i=4 (2018-01-08 00:00:00-05:00) ('AAPL', 'Open')=41.38, ('AAPL', 'High')=41.68, ('AAPL', 'Low')=41.28, ('AAPL', 'Close')=41.38, ('AAPL', 'Volume')=82271200.0>

# self.data.df
                            Open   High    Low  Close     Volume
dt                                                              
2018-01-02 00:00:00-05:00  40.39  40.90  40.18  40.89  102223600
2018-01-03 00:00:00-05:00  40.95  41.43  40.82  40.88  118071600
2018-01-04 00:00:00-05:00  40.95  41.18  40.85  41.07   89738400
2018-01-05 00:00:00-05:00  41.17  41.63  41.08  41.54   94640000
2018-01-08 00:00:00-05:00  41.38  41.68  41.28  41.38   82271200

# self.data.Close
[40.89 40.88 41.07 41.54 41.38]

# self.data["Close"]
[40.89 40.88 41.07 41.54 41.38]

# self.data.Close.df
dt
2018-01-02 00:00:00-05:00    40.89
2018-01-03 00:00:00-05:00    40.88
2018-01-04 00:00:00-05:00    41.07
2018-01-05 00:00:00-05:00    41.54
2018-01-08 00:00:00-05:00    41.38
Name: AAPL, dtype: float64

# self.data["Close"].df
dt
2018-01-02 00:00:00-05:00    40.89
2018-01-03 00:00:00-05:00    40.88
2018-01-04 00:00:00-05:00    41.07
2018-01-05 00:00:00-05:00    41.54
2018-01-08 00:00:00-05:00    41.38
Name: AAPL, dtype: float64

# self.data.df["Close"]
dt
2018-01-02 00:00:00-05:00    40.89
2018-01-03 00:00:00-05:00    40.88
2018-01-04 00:00:00-05:00    41.07
2018-01-05 00:00:00-05:00    41.54
2018-01-08 00:00:00-05:00    41.38
Name: Close, dtype: float64

# self.data.Close[-1]
41.38

# self.data.Close.df.iloc[-1]
41.38

# self.data["Close"][-1]
41.38

# self.data["Close"].df[-1]
41.38

# self.data.df["Close"][-1]
41.38
```

Please refer to [Backtesting.py](https://kernc.github.io/backtesting.py/) for more details as how to write a strategy for single asset.