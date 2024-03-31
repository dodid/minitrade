---
hide:
  - toc
---
# Compatibility with Backtesting.py

While Minitrade is largely compatible with [Backtesting.py](https://github.com/kernc/backtesting.py) for single asset strategy, some breaking changes still exist to support multi-asset strategies and to simplify usage.

## Differences

Here are a list of things that are incompatible:

1. The `data` input for `Backtest` should be a dataframe with 2-level column index. For example, 
    ```
    $ print(self.data)

               AAPL                              GOOG 
               Open  High  Low   Close Volume    Open  High  Low   Close Volume
    Date          
    2018-01-02 40.39 40.90 40.18 40.89 102223600 52.42 53.35 52.26 53.25 24752000
    2018-01-03 40.95 41.43 40.82 40.88 118071600 53.22 54.31 53.16 54.12 28604000
    2018-01-04 40.95 41.18 40.85 41.07 89738400 54.40 54.68 54.20 54.32 20092000
    2018-01-05 41.17 41.63 41.08 41.54 94640000 54.70 55.21 54.60 55.11 25582000
    2018-01-08 41.38 41.68 41.28 41.38 82271200 55.11 55.56 55.08 55.35 20952000
    ```
    For single asset, `data`, as a single level column DataFrame, still works. Since asset name is not specified, it's default to "Asset".
    ``` 
    $ print(self.data)

               Open  High  Low   Close Volume 
    Date          
    2018-01-02 40.39 40.90 40.18 40.89 102223600 
    2018-01-03 40.95 41.43 40.82 40.88 118071600 
    2018-01-04 40.95 41.18 40.85 41.07 89738400 
    2018-01-05 41.17 41.63 41.08 41.54 94640000 
    2018-01-08 41.38 41.68 41.28 41.38 82271200 
    ```
    Minitrade expects `Volume` data to be always avaiable. `data` should be consisted of OHLCV.

2. Calling `Strategy.next()` starts on the first bar when all data and indicators become available, in contrast to on the 2nd bar as implemented in the original library. **This will likely change backtest result.**

3. `Strategy.position` is no longer a property, but a function of signature
   ```python
   def position(ticker=None):
   ```
   `ticker` can be omitted in single asset strategies. `self.position` should be changed to `self.position()` in strategy code.

4. `Strategy.trades` is no longer a property, but a function of signature
   ```python
   def trades(ticker=None):
   ```
   `ticker` can be omitted in single asset strategies. `self.trades` should be changed to `self.trades()` in strategy code.

5. `Trade.value` is no longer the absolute value of the trade. It's a signed value that can be negative for short trade. 

6. `Strategy.orders` returns a plain `List[Order]`. If `self.orders.cancel()` is used with a semantic meaning of "Cancel all non-contingent (i.e. SL/TP) orders.", it should be replaced with the following explicitly.

    ```python
    for order in self.orders:
        if not order.is_contingent:
            order.cancel()
    ```

7. Pandas DataFrame and Series have richer APIs to work with, and therefore, are preferred over Numpy arrays when defining indicators in `Strategy.init()`. `Strategy.I()` returns Pandas objects. `Strategy._indicators` holds a list of Pandas objects rather than Numpy arrays. Variables derived from an indicator are no longer automatically added as indicators. Explicitly wrap in `I()` to define new indicators. For example,

    ```python
    self.sma = self.I(SMA, self.data.Close, 10)             # defines an indicator
    self.not_a_indicator = np.cumsum(self.sma * 5 + 1)      # not an indicator
    self.a_indicator = self.I(np.cumsum(self.sma * 5 + 1))  # another indicator
    ```

8. `Strategy.I()` takes both function and value arguments. With `self.data.ta` accessor, it's easier to precompute the indicator value then wrap it in `I()` than to present as a function call. For example, the following is equivalent, but the former is visually simpler.

    ```python
    self.sma = self.I(self.data.ta.sma(self.n_sma))         #1
    self.sma = self.I(ta.SMA, self.data.Close, self.n_sma)  #2
    ```

9. Since indexing of Numpy array is much faster than that of Pandas objects, indicators in `Strategy.next()` context are returned as Numpy arrays by default. Use `.df` to access the Pandas value, either as DataFrame or Series, of the indicator. It's the caller's reponsibility to keep track of which exact type should be returned. `.s` accessor is also available but only as a syntax suger to return a Series. If the actual data is a DataFrame, `.s` throws a `ValueError`.


8. Comparing indicators directly is no longer supported. Be explicit of what you want to compare, for example, 

    ```python
    if self.data.Close > self.sma:          # no longer work
    if self.data.Close[-1] > self.sma[-1]:  # use indexing explicitly
    ```
    Similarly, use indicators directly in a boolean context is not longer supported. 
    ```python
    if self.doji:                              # no longer work
    if bool(self.doji[-1]):                 # use indexing and conversion explicitly
    ```

9.  `Backtest` now has an extra argument `fail_fast` default to `True`, which means backtest will abort whenever an error occurs, e.g. cash is not enough to cover intended orders. This is to detect issues early in a live trading environment. If it's not desired in backtesting, set it to `False`.

10. There are some changes to the default values for plotting. `plot_volume` and `superimpose` are `False` by default. Minitrade doesn't try to guess where to plot the indicators. So to overlay the indicators on the main chart, set `overlay=True` explicitly.

## Examples

The following notebooks from [Backtesting.py](https://github.com/kernc/backtesting.py) are adapted to show how to make them work with Minitrade. Only the code part is changed, the text content may or may not apply to Minitrade. 

- [Quick Start User Guide](examples/Quick%20Start%20User%20Guide.ipynb)
- [Strategies Librry](examples/Strategies%20Library.ipynb)
- [Multiple Time Frames](examples/Multiple%20Time%20Frames.ipynb)
- [Parameter Heatmap & Optimization](examples/Parameter%20Heatmap%20&%20Optimization.ipynb)
- [Trading with Machine Learning](examples/Trading%20with%20Machine%20Learning.ipynb)