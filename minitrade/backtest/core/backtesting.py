"""
Core framework data structures.
Objects from this module can also be imported from the top-level
module directly, e.g.

    from minitrade.backtest import Backtest, Strategy
"""
import functools
import multiprocessing as mp
import os
import sys
import traceback
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from datetime import datetime
from functools import lru_cache, partial
from itertools import chain, compress, product, repeat
from math import copysign
from numbers import Number
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng

try:
    from tqdm.auto import tqdm as _tqdm
    _tqdm = partial(_tqdm, leave=False)
except ImportError:
    def _tqdm(seq, **_):
        return seq

from ._plotting import plot  # noqa: I001
from ._stats import compute_stats
from ._util import _as_str, _Data, _Indicator, try_

__pdoc__ = {
    'Strategy.__init__': False,
    'Order.__init__': False,
    'Position.__init__': False,
    'Trade.__init__': False,
}


class Allocation:
    '''The `Allocation` class manages the allocation of values among different assets in a portfolio. It provides 
    methods for creating and managing asset buckets, assigning weights to assets, and merging the weights into the 
    parent allocation object.

    `Allocation` is not meant to be instantiated directly. Instead, it is created automatically when a new
    `Strategy` object is created. The `Allocation` object is accessed through the `Strategy.alloc` property.

    The `Allocation` object is used as an input to the `Strategy.rebalance()` method to rebalance the portfolio according
    to the current weight allocation.

    `Allocation` has the following notable properties:

    - `tickers`: A list of tickers representing the asset space in the allocation.
    - `weights`: The weight allocation to be used in the current rebalance cycle.
    - `previous_weights`: The weight allocation used in the previous rebalance cycle.
    - `unallocated`: Unallocated equity weight, i.e. 1 minus the sum of weights already allocated to assets.
    - `bucket`: Bucket accessor for weight allocation.

    `Allocation` provides two ways to assign weights to assets:

    1. Explicitly assign weights to assets using `Allocation.weights` property. 

        It's possible to assign weights to individual asset or to all assets in the asset space as a whole. Not all weights 
        need to be specified. If an asset is not assigned a weight, it will have a weight of 0.

        Example:
        ```python
        # Assign weight to individual asset
        strategy.alloc.weights['A'] = 0.5

        # Assign weight to all assets
        strategy.alloc.weights = pd.Series([0.1, 0.2, 0.3], index=['A', 'B', 'C'])
        ```

    2. Use `Bucket` to assign weights to logical groups of assets, then merge the weights into the parent allocation object.

        A `Bucket` is a container that groups assets together and provieds methods for weight allocation. Assets can be added
        to the bucket by appending lists or filtering conditions. Weights can be assigned to the assets in the bucket using
        different allocation methods. Multiple buckets can be created for different groups of assets. Once the weight 
        allocation is done at bucket level , the weights of the buckets can be merged into those of the parent allocation object.

        Example:
        ```python
        # Create a bucket and add assets to it
        bucket = strategy.alloc.bucket['bucket1']
        bucket.append(['A', 'B', 'C'])

        # Assign weights to the assets in the bucket
        bucket.weight_explicitly([0.1, 0.2, 0.3])

        # Merge the bucket into the parent allocation object
        bucket.apply('update')
        ```

    The state of the `Allocation` object is managed by the `Strategy` object across rebalance cycles. A rebalance
    cycle involves:

    1. Initializing the weight allocation at the beginning of the cycle by calling either `Allocation.assume_zero()` 
    to reset all weights to zero or `Allocation.assume_previous()` to inherit the weights from the previous cycle. This
    must be done before any weight allocation attempts.
    2. Adjusting the weight allocation using either explicitly assignment or `Bucket` method.
    3. Calling `Strategy.rebalance()` to rebalance the portfolio according to the current allocation plan.

    After each rebalance cycle, the weight allocation is reset, and the process starts over. At any point, the weight
    allocation from the previous cycle can be accessed using the `previous_weights` property. 

    A rebalance cycle is not necessarily equal to the simulation time step. For example, simulation can be done at 
    daily frequency, while the portfolio is rebalanced every month. In this case, the weight allocation is maintained 
    across multiple time steps until the next time `Strategy.rebalance()` is called.

    Example:
    ```python
    class MyStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            # Initialize the weight allocation
            self.alloc.assume_zero()

            # Adjust the weight allocation
            self.alloc.bucket['equity'].append(['A', 'B', 'C']).weight_equally(sum_=0.4).apply('update')
            self.alloc.bucket['bond'].append(['D', 'E']).weight_equally(sum=_0.4).apply('update')
            self.alloc.weights['gold'] = self.alloc.unallocated

            # Rebalance the portfolio
            self.rebalance()
    ```
    '''

    class Bucket:
        '''`Bucket` is a container that groups assets together and applies weight allocation among them.
        A bucket is associated with a parent allocation object, while the allocation object can be
        associated with multiple buckets.

        Assets in a bucket are identified by their tickers. They are unique within the bucket, but can be
        repeated in different buckets. 

        Using `Bucket` for weight allocation takes 3 steps: 

        1. Assets are added to the bucket by appending lists or filtering conditions. The rank of the assets 
        in the bucket is preserved and can be used to assign weights. 
        2. Weights are assigned to the assets using different allocation methods. 
        3. Once the weight allocation at bucket level is done, the weights of the bucket can be merged into 
        those of the parent allocation object.
        '''

        def __init__(self, alloc: 'Allocation') -> None:
            self._alloc = alloc
            self._tickers = []
            self._weights = None

        @property
        def tickers(self) -> list:
            '''Assets in the bucket. This is a read-only property.'''
            return self._tickers.copy()

        @property
        def weights(self) -> pd.Series:
            '''Weights of the assets in the bucket. This is only available after weight allocation is done
            by calling `Bucket.weight_*()` methods. This is a read-only property.'''
            assert (self._weights >= 0).all(), 'Weight should be non-negative.'
            assert self._weights.sum(
            ) <= 1, f'Total weight should be less than or equal to 1. Got {self._weights.sum()}'
            return self._weights.copy()

        def append(self, ranked_list: list | pd.Series, *conditions: list | pd.Series) -> 'Allocation.Bucket':
            '''Add assets that are in the ranked list to the end of the bucket.

            `ranked_list` can be specified in three ways:

            1. A list of assets or anything list-like, all items will be added.
            2. A boolean Series with assets as the index and a True value to indicate the asset should be added.
            3. A non-boolean Series with assets as the index and all assets in the index will be added.

            The rank of the assets is determined by its order in the list or in the index. The rank of the assets 
            in the bucket is preserved. If an asset is already in the bucket, its rank in bucket will not be affected
            by appending new list to the bucket, even if the asset is ranked differently in the new list.

            Multiple conditions can be specified as filters to exclude certain assets in the ranked list from being 
            added. Assets must satisfy all the conditions in order to be added to the bucket.

            `conditions` can be specified in the same way as `ranked_list`, only that the asset order in a condition
            is not important.

            Example:
            ```python
            # Append 'A' and 'B' to the bucket
            bucket.append(['A', 'B'])

            # Append 'A' and 'C' to the bucket
            bucket.append(pd.Series([True, False, True], index=['A', 'B', 'C']))

            # Append 'C' to the bucket
            bucket.append(pd.Series([1, 2, 3], index=['A', 'B', 'C']).nlargest(2), pd.Series([1, 2, 3], index=['A', 'B', 'C']) > 2)
            ```

            Args:
                ranked_list: A list of assets or a Series of assets to be added to the bucket.
                conditions: A list of assets or a Series of assets to be used as conditions to filter the assets.
            '''
            list_and_conditions = [ranked_list] + list(conditions)
            candidates = {}
            for item in list_and_conditions:
                item = [index for index, value in item.items() if not isinstance(
                    value, bool) or value] if isinstance(item, pd.Series) else list(item)
                for x in item:
                    candidates[x] = candidates.get(x, 0) + 1
            candidates = [x for x in candidates if candidates[x] == len(list_and_conditions)]
            self._tickers.extend([x for x in candidates if x not in self._tickers])
            return self

        def remove(self, *conditions: list | pd.Series) -> 'Allocation.Bucket':
            '''Remove assets that satisify all the given conditions from the bucket.

            `conditions` can be specified in three ways:

            1. A list of assets or anything list-like, all assets will be removed.
            2. A boolean Series with assets as the index and a True value to indicate the asset should be removed.
            3. A non-boolean Series with assets as the index and all assets in the index will be removed.

            Example:
            ```python
            # Remove 'A' and 'B' from the bucket
            bucket.remove(['A', 'B'])

            # Remove 'A' and 'C' from the bucket
            bucket.remove(pd.Series([True, False, True], index=['A', 'B', 'C']))

            # Remove 'A' and 'B' from the bucket
            bucket.remove(pd.Series([1, 2, 3], index=['A', 'B', 'C']).nsmallest(2))

            # Remove 'B' from the bucket
            bucket.remove(pd.Series([1, 2, 3], index=['A', 'B', 'C']) > 1, pd.Series([1, 2, 3], index=['A', 'B', 'C']) < 3)
            ```
            Args:
                conditions: A list of assets or a Series of assets to be used as conditions to filter the assets.
            '''
            if len(conditions) == 0:
                return
            candidates = {}
            for item in conditions:
                item = [index for index, value in item.items() if not isinstance(
                    value, bool) or value] if isinstance(item, pd.Series) else list(item)
                for x in item:
                    candidates[x] = candidates.get(x, 0) + 1
            self._tickers = [x for x in self._tickers if candidates.get(x, 0) < len(conditions)]
            return self

        def trim(self, limit: int) -> 'Allocation.Bucket':
            '''Trim the bucket to a maximum number of assets.

            Args:
                limit: Maximum number of assets should be included
            '''
            self._tickers = self._tickers[:limit]
            return self

        def weight_explicitly(self, weight: float | list | pd.Series) -> 'Allocation.Bucket':
            '''Assign weights to the assets in the bucket.

            `weight` can be specified in three ways:

            1. A single weight should be assigned to all assets in the bucket.
            2. A list of weights should be assigned to the assets in the bucket in rank order. If more weights are provided than the number of assets in the bucket, the extra weights are ignored. If fewer weights are provided, the remaining assets will be assigned a weight of 0.
            3. A Series with assets as the index and the weight as the value. If no weight is provided for an asset, it will be assigned a weight of 0. If a weight is provided for an asset that is not in the bucket, it will be ignored.

            Example:
            ```python
            bucket.append(['A', 'B', 'C']).weight_explicitly(0.2)
            bucket.append(['A', 'B', 'C']).weight_explicitly([0.1, 0.2, 0.3])
            bucket.append(['A', 'B', 'C']).weight_explicitly(pd.Series([0.1, 0.2, 0.3], index=['A', 'B', 'C']))
            ```
            Args:
                weight: A single value, a list of values or a Series of weights.
            '''
            if len(self._tickers) == 0:
                self._weights = pd.Series()
            elif isinstance(weight, Number):
                assert 0 <= weight * len(self._tickers) <= 1, 'Total weight should be within [0, 1].'
                self._weights = pd.Series(weight, index=self._tickers)
            elif isinstance(weight, list):
                assert all(0 <= x <= 1 for x in weight), 'Weight should be non-negative.'
                assert sum(weight) <= 1, 'Total weight should be less than or equal to 1.'
                weight = weight[:len(self._tickers)]
                weight.extend([0.] * (len(self._tickers) - len(weight)))
                self._weights = pd.Series(weight, index=self._tickers)
            elif isinstance(weight, pd.Series):
                assert (weight >= 0).all(), 'Weight should be non-negative.'
                assert weight.sum() <= 1, 'Total weight should be less than or equal to 1.'
                weight = weight[weight.index.isin(self._tickers)]
                self._weights = pd.Series(0., index=self._tickers)
                self._weights.loc[weight.index] = weight
            else:
                raise ValueError('Weight should be a single value, a list of values or a Series of weights.')
            return self

        def weight_equally(self, sum_: float = None) -> 'Allocation.Bucket':
            '''Allocate equity value equally to the assets in the bucket.

            `sum_` should be between 0 and 1, with 1 means 100% of value should be allocated.

            Example:
            ```python
            bucket.append(['A', 'B', 'C']).weight_equally(0.5)
            ```

            Args:
                sum_: Total weight that should be allocated. 
            '''
            assert sum_ is None or 0 <= sum_ <= 1, 'Total weight should be within [0, 1].'
            if sum_ is None:
                sum_ = self._alloc.unallocated
            if len(self._tickers) == 0:
                self._weights = pd.Series()
            else:
                self._weights = pd.Series(1 / len(self._tickers), index=self._tickers) * sum_
            return self

        def weight_proportionally(self, relative_weights: list, sum_: float = None) -> 'Allocation.Bucket':
            '''Allocate equity value proportionally to the assets in the bucket.

            `sum_` should be between 0 and 1, with 1 means 100% of value should be allocated.

            Example:
            ```python
            bucket.append(['A', 'B', 'C']).weight_proportionally([1, 2, 3], 0.5)
            ```

            Args:
                relative_weights: A list of relative weights. The length of the list should be the same as the number of assets in the bucket.
                sum_: Total weight that should be allocated. 
            '''
            assert len(relative_weights) == len(
                self._tickers), f'Length of relative_weight {len(relative_weights)} does not match number of assets {len(self._tickers)}'
            assert all(x >= 0 for x in relative_weights), 'Relative weights should be non-negative.'
            assert sum_ is None or 0 <= sum_ <= 1, 'Total weight should be within [0, 1].'
            if sum_ is None:
                sum_ = self._alloc.unallocated
            if len(self._tickers) == 0:
                self._weights = pd.Series()
            else:
                self._weights = pd.Series(relative_weights, index=self._tickers) / sum(relative_weights) * sum_
            return self

        def apply(self, method: str = 'update') -> 'Allocation.Bucket':
            '''Apply the weight allocation to the parent allocation object.

            `method` controls how the bucket weight allocation should be merged into the parent allocation object.

            When `method` is `update`, the weights of assets in the bucket will update the weights of the same assets
            in the parent allocation object. If an asset is not in the bucket, its weight in the parent allocation object 
            will not be changed. This is the default method.

            When `method` is `overwrite`, the weights of the parent allocation object will be replaced by the weights of the 
            assets in the bucket or set to 0 if the asset is not in the bucket.

            When `method` is `accumulate`, the weights of the assets in the bucket will be added to the weights of the same 
            assets, while the weights of the assets not in the bucket will remain unchanged.

            If the bucket is empty, no change will be made to the parent allocation object.

            Note that no validation is performed on the weights of the parent allocation object after the bucket weight
            is merged. It is the responsibility of the user to ensure the final weights are valid before use.

            Args:
                method: Method to merge the bucket into the parent allocation object. 
                    Available methods are 'update', 'overwrite', 'accumulate'.
            '''
            if self._weights is None:
                raise RuntimeError('Bucket.weight_*() should be called before apply()')
            if self.weights.empty:
                return self
            index = self.weights.index
            if method == 'update':
                self._alloc.weights.loc[index] = self.weights
            elif method == 'overwrite':
                self._alloc.weights.loc[:] = 0.
                self._alloc.weights.loc[index] = self.weights
            elif method == 'accumulate':
                self._alloc.weights.loc[index] = self._alloc.weights.loc[index] + self.weights
            else:
                raise ValueError(f'Invalid method {method}')
            return self

        def __len__(self) -> int:
            return len(self._tickers)

        def __iter__(self):
            return iter(self._tickers)

        def __eq__(self, other):
            if isinstance(other, pd.Series):
                return self._weights.equals(other)
            elif isinstance(other, list):
                return self._tickers == other
            else:
                return False

        def __repr__(self) -> str:
            return f'Bucket(tickers={self._tickers})'

    class BucketGroup:
        def __init__(self, alloc: 'Allocation') -> None:
            self._alloc = alloc
            self._buckets = {}

        def clear(self) -> None:
            self._buckets.clear()

        def __getitem__(self, name: str) -> 'Allocation.Bucket':
            if name not in self._buckets:
                self._buckets[name] = Allocation.Bucket(self._alloc)
            return self._buckets[name]

        def __iter__(self):
            return iter(self._buckets)

        def __len__(self) -> int:
            return len(self._buckets)

    def __init__(self, tickers: list) -> None:
        self._tickers = tickers
        self._previous_weights = pd.Series(0., index=tickers)
        self._weights = None
        self._bucket_group = Allocation.BucketGroup(self)

    def _after_assume(func):
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            if self._weights is None:
                raise RuntimeError('"Allocation.assume_*()" must be called first.')
            return func(self, *args, **kwargs)
        return inner

    @property
    def tickers(self) -> list:
        '''Assets representing the asset space. This is a read-only property'''
        return self._tickers.copy()

    @property
    @_after_assume
    def bucket(self) -> BucketGroup:
        '''`bucket` provides access to a dictionary of buckets.

        A bucket can be accessed with a string key. If the bucket does not exist, one will be created automatically.

        Buckets are cleared after each rebalance cycle.

        Example:
        ```python
        # Access the bucket named 'equity'
        bucket = strategy.alloc.bucket['equity']
        ```
        '''
        return self._bucket_group

    @property
    @_after_assume
    def weights(self) -> pd.Series:
        '''Current weight allocation. Weight should be non-negative and the total weight should be less than or equal to 1.

        It's possible to assign weights to individual asset or to all assets in the asset space as a whole. When assigning
        weights as a whole, only non-zero weights need to be specified, and other weights are assigned zero automatically.

        Example:
        ```python
        # Assign weight to individual asset
        strategy.alloc.weights['A'] = 0.5

        # Assign weight to all assets
        strategy.alloc.weights = pd.Series([0.1, 0.2, 0.3], index=['A', 'B', 'C'])
        ```
        '''
        assert self._weights.index.to_list() == self._tickers, 'Weight index should be the same as the asset space.'
        assert (self._weights >= 0).all(), 'Weight should be non-negative.'
        assert self._weights.sum() <= 1, f'Total weight should be less than or equal to 1. Got {self._weights.sum()}'
        return self._weights

    @weights.setter
    @_after_assume
    def weights(self, value: pd.Series) -> None:
        assert (value >= 0).all(), 'Weight should be non-negative.'
        assert value.sum() <= 1, f'Total weight should be less than or equal to 1. Got {value.sum()}'
        self._weights.loc[:] = 0.
        self._weights.loc[value.index] = value

    @property
    def previous_weights(self) -> pd.Series:
        '''Previous weight allocation. This is a read-only property.'''
        return self._previous_weights.copy()

    def assume_zero(self):
        '''Assume all assets have zero weight to begin with in a new rebalance cycle. 
        '''
        self._weights = pd.Series(0., index=self.tickers)

    def assume_previous(self):
        '''Assume all assets inherit the same weight as used in the previous rebalance cycle.
        '''
        self._weights = self.previous_weights.copy()

    @property
    @_after_assume
    def unallocated(self) -> float:
        '''Unallocated equity weight. It's the remaining weight that can be allocated to assets. This is a read-only property.'''
        allocated = self._weights.abs().sum()
        assert allocated <= 1, f'Total weight should be less than or equal to 1. Got {allocated}'
        return 1. - allocated

    @_after_assume
    def normalize(self):
        '''Normalize the weight allocation so that the sum of weights equals 1.'''
        self._weights = self._weights / self._weights.abs().sum()
        return self.weights

    @property
    @_after_assume
    def modified(self):
        '''True if weight allocation is changed from previous values.'''
        return not self.weights.equals(self.previous_weights)

    def _next(self):
        '''Prepare for the next rebalance cycle. This is called after each call to `Strategy.rebalance()`.
        '''
        self._previous_weights = self._weights.copy()
        self._weights = None
        self._bucket_group.clear()

    def _clear(self):
        '''Clear the weight allocation and buckets.
        '''
        self._previous_weights = pd.Series(0., index=self._tickers)
        self._weights = None
        self._bucket_group.clear()


class Strategy(ABC):
    """
    A trading strategy base class. Extend this class and
    override methods
    `minitrade.backtest.core.backtesting.Strategy.init` and
    `minitrade.backtest.core.backtesting.Strategy.next` to define
    your own strategy.
    """

    def __init__(self, broker, data, params):
        self._indicators = []
        self._broker: _Broker = broker
        self._data: _Data = data
        self._params = self._check_params(params)
        self._alloc = Allocation(data.tickers)
        self._data_index = data.index.copy()
        self._records = {}
        self._start_on_day = 0

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'

    def __str__(self):
        params = ','.join(f'{i[0]}={i[1]}' for i in zip(self._params.keys(),
                                                        map(_as_str, self._params.values())))
        if params:
            params = '(' + params + ')'
        return f'{self.__class__.__name__}{params}'

    def _check_params(self, params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing parameter '{k}'."
                    "Strategy class should define parameters as class variables before they "
                    "can be optimized or run with.")
            setattr(self, k, v)
        return params

    def I(self,  # noqa: E743
          funcval: Union[pd.DataFrame, pd.Series, Callable], *args,
          name=None, plot=True, overlay=None, color=None, scatter=False,
          ** kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Declare an indicator. An indicator is just an array of values,
        but one that is revealed gradually in
        `minitrade.backtest.core.backtesting.Strategy.next` much like
        `minitrade.backtest.core.backtesting.Strategy.data` is.
        Returns DataFrame in `init()` and `np.ndarray` of indicator values in `next()`.

        `funcval` is either a function that returns the indicator array(s) of
        same length as `minitrade.backtest.core.backtesting.Strategy.data`, or
        the indicator array(s) itself as a DataFrame, Series, or arrays.

        In the plot legend, the indicator is labeled with function name, 
        DataFrame column name, or Series name, unless `name` overrides it.

        If `plot` is `True`, the indicator is plotted on the resulting
        `minitrade.backtest.core.backtesting.Backtest.plot`.

        If `overlay` is `True`, the indicator is plotted overlaying the
        price candlestick chart (suitable e.g. for moving averages).
        If `False`, the indicator is plotted standalone below the
        candlestick chart. 

        `color` can be string hex RGB triplet or X11 color name.
        By default, the next available color is assigned.

        If `scatter` is `True`, the plotted indicator marker will be a
        circle instead of a connected line segment (default).

        Additional `*args` and `**kwargs` are passed to `func` and can
        be used for parameters.

        For example, using simple moving average function from TA-Lib:

            def init():
                self.sma = self.I(ta.SMA, self.data.Close, self.n_sma)
        """
        if callable(funcval):
            if name is None:
                params = ','.join(filter(None, map(_as_str, chain(args, kwargs.values()))))
                func_name = _as_str(funcval)
                name = (f'{func_name}({params})' if params else f'{func_name}')
            else:
                name = name.format(*map(_as_str, args),
                                   **dict(zip(kwargs.keys(), map(_as_str, kwargs.values()))))
            try:
                value = funcval(*args, **kwargs)
            except Exception as e:
                raise RuntimeError(f'Indicator "{funcval}" error') from e
        else:
            value = funcval

        if isinstance(value, (pd.DataFrame, pd.Series)):
            if not value.index.equals(self._data.index):
                raise ValueError(
                    'Indicators of pd.DataFrame or pd.Series must have the same index as'
                    f' `data` (data shape: {len(self._data)}; indicator shape: {len(value)}.\n'
                    f'`data` index: {self._data.index}\n'
                    f'Indicator index: {value.index}\n')
            value = value.copy()
        else:
            if value is not None:
                value = try_(lambda: np.asarray(value, order='C'), None)
            is_arraylike = bool(value is not None and value.shape)

            # Optionally flip the array if the long side of array is not on the 1st dimension
            if is_arraylike and np.argmin(value.shape) == 0:
                value = value.T

            if not is_arraylike or not 1 <= value.ndim <= 2 or value.shape[0] != len(self._data):
                raise ValueError(
                    'Indicators of numpy.ndarray must have the same '
                    f'length as `data` (data shape: {len(self._data)}; indicator "{name}" '
                    f'shape: {getattr(value, "shape" , "")}, returned value: {value})')
            elif value.ndim == 1:
                value = pd.Series(value, index=self._data.index, name=name)
            else:
                value = pd.DataFrame(value, index=self._data.index)

        # Use an experimental feature to save DataFrame/Series metadata
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.attrs.html
        value.attrs.update({'name': name, 'plot': plot, 'overlay': overlay,
                           'color': color, 'scatter': scatter, **kwargs})
        self._indicators.append(value)
        return value

    @abstractmethod
    def init(self):
        """
        Initialize the strategy.
        Override this method.
        Declare indicators (with `minitrade.backtest.core.backtesting.Strategy.I`).
        Precompute what needs to be precomputed or can be precomputed
        in a vectorized fashion before the strategy starts.

        If you extend composable strategies from `minitrade.backtest.core.backtesting.lib`,
        make sure to call: `super().init()`
        """

    @abstractmethod
    def next(self):
        """
        Main strategy runtime method, called as each new
        `minitrade.backtest.core.backtesting.Strategy.data`
        instance (row; full candlestick bar) becomes available.
        This is the main method where strategy decisions
        upon data precomputed in `minitrade.backtest.core.backtesting.Strategy.init`
        take place.

        If you extend composable strategies from `minitrade.backtest.core.backtesting.lib`,
        make sure to call: `super().next()`
        """

    class __FULL_EQUITY(float):  # noqa: N801
        def __repr__(self): return '.9999'
    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    def buy(self, *,
            ticker: str = None,
            size: float = _FULL_EQUITY,
            limit: Optional[float] = None,
            stop: Optional[float] = None,
            sl: Optional[float] = None,
            tp: Optional[float] = None,
            tag: object = None):
        """
        Place a new long order. For explanation of parameters, see `Order` and its properties.

        For single asset strategy, `ticker` can be left as None.

        See `Position.close()` and `Trade.close()` for closing existing positions.

        See also `Strategy.sell()`.
        """
        assert 0 < size < 1 or round(size) == size, \
            "size must be a positive fraction of equity, or a positive whole number of units"
        return self._broker.new_order(ticker, size, limit, stop, sl, tp, tag)

    def sell(self, *,
             ticker: str = None,
             size: float = _FULL_EQUITY,
             limit: Optional[float] = None,
             stop: Optional[float] = None,
             sl: Optional[float] = None,
             tp: Optional[float] = None,
             tag: object = None):
        """
        Place a new short order. For explanation of parameters, see `Order` and its properties.

        For single asset strategy, `ticker` can be left as None.

        See also `Strategy.buy()`.

        .. note::
            If you merely want to close an existing long position,
            use `Position.close()` or `Trade.close()`.
        """
        assert 0 < size < 1 or round(size) == size, \
            "size must be a positive fraction of equity, or a positive whole number of units"
        return self._broker.new_order(ticker, -size, limit, stop, sl, tp, tag)

    def rebalance(self, force: bool = False, rtol: float = 0.01, atol: int = 0, cash_reserve: float = 0.1):
        """
        Rebalance the portfolio according to the current weight allocation.

        If the weight allocation is not changed from the previous cycle, the rebalance is skipped. This behavior can be
        overridden by setting `force` to `True`, which will force rebalance even if the weight allocation is unchanged.
        This is useful when the actual portfolio value deviates from the target value due to price changes and should 
        be corrected.

        When a rebalance should be performed, the difference between the target and actual portfolio, defined as the sum 
        of absolute difference of individual assets, is calculated. If the difference is too small compared to the
        relative tolerance `rtol` or the absolute tolerance `atol`, the rebalance is again skipped. This can be used
        to avoid unnecessary transaction cost. An exception is when the target weight of an asset is zero, in which case 
        the position of the asset, if exists, is always closed.

        `cash_reserve` is the ratio of total equity reserved as cash to account for order quantity rounding and sudden
        price changes between order placement and execution. It is recommended to set this value to a small positive
        number to avoid order rejection due to insufficient cash. The minimum value may depend on the volatility of the 
        assets.

        Args:
            force: If True, rebalance will be performed even if the current weight allocation
                is not changed from the previous.
            rtol: Relative tolerance of the total absolute value difference between current 
                and previous allocation vs. total portfolio value. If the difference is smaller 
                than `rtol`, rebalance will not be performed.
            atol: Absolute tolerance of the total absolute value difference between current 
                and previous allocation. If the difference is smaller than `atol`, rebalance 
                will not be performed.
            cash_reserve: Ratio of total equity reserved as cash to account for order 
                quantity rounding and sudden price changes between order placement and
                execution. 
        """
        self._broker.rebalance(alloc=self._alloc, force=force, rtol=rtol, atol=atol, cash_reserve=cash_reserve)

    def record(self, name: str = None, plot: bool = True, overlay: bool = None, color: str = None, scatter: bool = False, **kwargs):
        """
        Record arbitrary key-value pairs as time series. This can be used for diagnostic
        data collection or for plotting custom data. 

        Values to be recorded should be passed as keyword arguments. The value can be a scalar, a dictionary, or a
        pandas Series. If a dictionary or a Series is passed, its keys will be used as names for time series.

        Example:
        ```python
        # Record a scalar value
        self.record(my_key=42)

        # Record a dictionary
        self.record(my_dict={'a': 1, 'b': 2})

        # Record a pandas Series
        self.record(my_series=pd.Series({'a': 1, 'b': 2}))
        ```

        Args:
            name: Name of the time series. If not provided, the name will be the same as the keyword argument.
            plot: If True, the time series will be plotted on the resulting `minitrade.backtest.core.backtesting.Backtest.plot`.
            overlay: If True, the time series will be plotted overlaying the price candlestick chart. If False, the time series
                will be plotted standalone below the candlestick chart.
            color: Color of the time series. If not provided, the next available color will be assigned.
            scatter: If True, the plotted time series marker will be a circle instead of a connected line segment.
        """
        for k, v in kwargs.items():
            if isinstance(v, dict) or isinstance(v, pd.Series):
                v = dict(v)
                if k not in self._records:
                    self._records[k] = pd.DataFrame(index=self._data_index, columns=v.keys())
                self._records[k].loc[self._broker.now, list(v.keys())] = list(v.values())
            else:
                if k not in self._records:
                    self._records[k] = pd.Series(index=self._data_index)
                self._records[k].iloc[len(self._data)-1] = v
            self._records[k].name = name or k
            self._records[k].attrs.update({'name': name or k, 'plot': plot, 'overlay': overlay,
                                           'color': color, 'scatter': scatter})

    @property
    def equity(self) -> float:
        """Current account equity (cash plus assets)."""
        return self._broker.equity()

    @property
    def data(self) -> _Data:
        """
        Price data, roughly as passed into
        `minitrade.backtest.core.backtesting.Backtest.__init__`,
        but with two significant exceptions:

        * `data` is _not_ a DataFrame, but a custom structure
          that serves customized numpy arrays for reasons of performance
          and convenience. Besides OHLCV columns, `.index` and length,
          it offers `.pip` property, the smallest price unit of change.
        * Within `minitrade.backtest.core.backtesting.Strategy.init`, `data` arrays
          are available in full length, as passed into
          `minitrade.backtest.core.backtesting.Backtest.__init__`
          (for precomputing indicators and such). However, within
          `minitrade.backtest.core.backtesting.Strategy.next`, `data` arrays are
          only as long as the current iteration, simulating gradual
          price point revelation. In each call of
          `minitrade.backtest.core.backtesting.Strategy.next` (iteratively called by
          `minitrade.backtest.core.backtesting.Backtest` internally),
          the last array value (e.g. `data.Close[-1]`)
          is always the _most recent_ value.
        * If you need data arrays (e.g. `data.Close`) to be indexed
          **Pandas Series or DataFrame**, you can call their `.df` accessor
          (e.g. `data.Close.df`). If you need the whole of data
          as a **DataFrame**, use `.df` accessor (i.e. `data.df`).
        """
        return self._data

    @property
    def storage(self) -> dict | None:
        """Storage is a dictionary for saving custom data across backtest runs
        when used in the context of automated trading in incremental mode. 

        If backtest finishes successfully, any modification to the dictionary 
        is persisted and can be accessed in future runs. If backtest fails due 
        to any error, the modification is not saved. If backtest runs in dryrun 
        mode, the modification is not saved.

        No storage is provided when trading in "strict" mode, in which case `storage` 
        is None. 
        """
        return self._broker._storage

    def position(self, ticker: str = None) -> 'Position':
        """Instance of `minitrade.backtest.core.backtesting.Position`.

        For single asset strategy, `ticker` can be left as None, which returns
        the position of the only asset.
        """
        ticker = ticker or self._data.the_ticker
        return self._broker.positions[ticker]

    @property
    def orders(self) -> 'List[Order]':
        """List of orders (see `Order`) waiting for execution."""
        return self._broker.orders

    def trades(self, ticker: str = None) -> 'Tuple[Trade, ...]':
        """List of active trades (see `Trade`)."""
        return tuple(self._broker.trades[ticker] if ticker else self._broker.all_trades)

    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        """List of settled trades (see `Trade`)."""
        return tuple(self._broker.closed_trades)

    @property
    def alloc(self) -> Allocation:
        """`Allocation` instance that manages the weight allocation among assets."""
        return self._alloc

    def start_on_day(self, n: int):
        """Hint to start the backtest on a specific day.

        This can be used to define a warm-up period, ensuring at least `n` days of data 
        are available when `next()` is called for the first time. 

        When the backtest starts depends both on `n` and on the availability of indicators. 
        If indicators are defined, the backtest will start when all indicators have 
        valid data or on the `n`-th day, whichever comes later.

        This method should be called in `init()`.

        Args:
            n: Day index to start on. Must be within [0, len(data)-1].
        """
        assert 0 <= n < len(self._data), f"day must be within [0, {len(self._data)-1}]"
        self._start_on_day = n

    @classmethod
    def prepare_data(cls, tickers: 'List[str]', start: str) -> pd.DataFrame | None:
        """Prepare data for trading.

        This class method can be overridden in a `Strategy` implementation to provide
        data for trading. The can be useful when the data is not provided externally
        and the strategy wants to bring its own data, e.g. from a database.

        Args:
            tickers: List of tickers to fetch data for.
            start: Start date of the data to fetch.

        Returns:
            A `pd.DataFrame` with 2-level columns as required by `Backtest()` or None.
        """
        return None


class Position:
    """
    Currently held asset position, available as
    `minitrade.backtest.core.backtesting.Strategy.position` within
    `minitrade.backtest.core.backtesting.Strategy.next`.
    Can be used in boolean contexts, e.g.

        if self.position():
            ...  # we have a position, either long or short
    """

    def __init__(self, broker: '_Broker', ticker: str):
        self.__broker = broker
        self.__ticker = ticker

    def __bool__(self):
        return self.size != 0

    @property
    def size(self) -> float:
        """Position size in units of asset. Negative if position is short."""
        return sum(trade.size for trade in self.__broker.trades[self.__ticker])

    @property
    def pl(self) -> float:
        """Profit (positive) or loss (negative) of the current position in cash units."""
        return sum(trade.pl for trade in self.__broker.trades[self.__ticker])

    @property
    def pl_pct(self) -> float:
        """Profit (positive) or loss (negative) of the current position in percent."""
        weights = np.abs([trade.size for trade in self.__broker.trades[self.__ticker]])
        weights = weights / weights.sum()
        pl_pcts = np.array([trade.pl_pct for trade in self.__broker.trades[self.__ticker]])
        return (pl_pcts * weights).sum()

    @property
    def is_long(self) -> bool:
        """True if the position is long (position size is positive)."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """True if the position is short (position size is negative)."""
        return self.size < 0

    def close(self, portion: float = 1.):
        """
        Close portion of position by closing `portion` of each active trade. See `Trade.close`.
        """
        for trade in self.__broker.trades[self.__ticker]:
            trade.close(portion)

    def __repr__(self):
        return f'<Position: {self.size} ({len(self.__broker.trades[self.__ticker])} trades)>'


class _OutOfMoneyError(Exception):
    pass


class Order:
    """
    Place new orders through `Strategy.buy()` and `Strategy.sell()`.
    Query existing orders through `Strategy.orders`.

    When an order is executed or [filled], it results in a `Trade`.

    If you wish to modify aspects of a placed but not yet filled order,
    cancel it and place a new one instead.

    All placed orders are [Good 'Til Canceled].

    [filled]: https://www.investopedia.com/terms/f/fill.asp
    [Good 'Til Canceled]: https://www.investopedia.com/terms/g/gtc.asp
    """

    def __init__(self, broker: '_Broker',
                 ticker: str,
                 size: float,
                 limit_price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 sl_price: Optional[float] = None,
                 tp_price: Optional[float] = None,
                 parent_trade: Optional['Trade'] = None,
                 entry_time: datetime = None,
                 tag: object = None):
        self.__broker = broker
        self.__ticker = ticker
        assert size != 0
        self.__size = size
        self.__limit_price = limit_price
        self.__stop_price = stop_price
        self.__sl_price = sl_price
        self.__tp_price = tp_price
        self.__parent_trade = parent_trade
        self.__entry_time = entry_time
        self.__tag = tag

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def __repr__(self):
        return f'<Order {self.__ticker} {{}}>'.format(', '.join(f'{param}={round(value, 5)}'
                                                                for param, value in (
                                                                    ('size', self.__size),
                                                                    ('limit', self.__limit_price),
                                                                    ('stop', self.__stop_price),
                                                                    ('sl', self.__sl_price),
                                                                    ('tp', self.__tp_price),
                                                                    ('contingent', self.is_contingent),
                                                                ) if value is not None))

    def cancel(self):
        """Cancel the order."""
        self.__broker.orders.remove(self)
        trade = self.__parent_trade
        if trade:
            if self is trade._sl_order:
                trade._replace(sl_order=None)
            elif self is trade._tp_order:
                trade._replace(tp_order=None)
            else:
                # XXX: https://github.com/kernc/backtesting.py/issues/251#issuecomment-835634984 ???
                assert False

    # Fields getters

    @property
    def ticker(self) -> str:
        return self.__ticker

    @property
    def size(self) -> float:
        """
        Order size (negative for short orders).

        If size is a value between 0 and 1, it is interpreted as a fraction of current
        available liquidity (cash plus `Position.pl` minus used margin).
        A value greater than or equal to 1 indicates an absolute number of units.
        """
        return self.__size

    @size.setter
    def size(self, size):
        """ Setter of order size """
        self.__size = size

    @property
    def limit(self) -> Optional[float]:
        """
        Order limit price for [limit orders], or None for [market orders],
        which are filled at next available price.

        [limit orders]: https://www.investopedia.com/terms/l/limitorder.asp
        [market orders]: https://www.investopedia.com/terms/m/marketorder.asp
        """
        return self.__limit_price

    @property
    def stop(self) -> Optional[float]:
        """
        Order stop price for [stop-limit/stop-market][_] order,
        otherwise None if no stop was set, or the stop price has already been hit.

        [_]: https://www.investopedia.com/terms/s/stoporder.asp
        """
        return self.__stop_price

    @property
    def sl(self) -> Optional[float]:
        """
        A stop-loss price at which, if set, a new contingent stop-market order
        will be placed upon the `Trade` following this order's execution.
        See also `Trade.sl`.
        """
        return self.__sl_price

    @property
    def tp(self) -> Optional[float]:
        """
        A take-profit price at which, if set, a new contingent limit order
        will be placed upon the `Trade` following this order's execution.
        See also `Trade.tp`.
        """
        return self.__tp_price

    @property
    def parent_trade(self):
        return self.__parent_trade

    @property
    def tag(self):
        """
        Arbitrary value (such as a string) which, if set, enables tracking
        of this order and the associated `Trade` (see `Trade.tag`).
        """
        return self.__tag

    __pdoc__['Order.parent_trade'] = False

    # Extra properties

    @property
    def is_long(self):
        """True if the order is long (order size is positive)."""
        return self.__size > 0

    @property
    def is_short(self):
        """True if the order is short (order size is negative)."""
        return self.__size < 0

    @property
    def is_contingent(self):
        """
        True for [contingent] orders, i.e. [OCO] stop-loss and take-profit bracket orders
        placed upon an active trade. Remaining contingent orders are canceled when
        their parent `Trade` is closed.

        You can modify contingent orders through `Trade.sl` and `Trade.tp`.

        [contingent]: https://www.investopedia.com/terms/c/contingentorder.asp
        [OCO]: https://www.investopedia.com/terms/o/oco.asp
        """
        return bool(self.__parent_trade)

    @property
    def entry_time(self) -> datetime:
        """Time of when the order is created."""
        return self.__entry_time


class Trade:
    """
    When an `Order` is filled, it results in an active `Trade`.
    Find active trades in `Strategy.trades` and closed, settled trades in `Strategy.closed_trades`.
    """

    def __init__(self, broker: '_Broker', ticker: str, size: int, entry_price: float, entry_bar, tag):
        self.__broker = broker
        self.__ticker = ticker
        self.__size = size
        self.__entry_price = entry_price
        self.__exit_price: Optional[float] = None
        self.__entry_bar: int = entry_bar
        self.__exit_bar: Optional[int] = None
        self.__sl_order: Optional[Order] = None
        self.__tp_order: Optional[Order] = None
        self.__tag = tag

    def __repr__(self):
        return f'<Trade size={self.__size} time={self.__entry_bar}-{self.__exit_bar or ""} ' \
               f'price={self.__entry_price}-{self.__exit_price or ""} pl={self.pl:.0f}' \
               f'{" tag="+str(self.__tag) if self.__tag is not None else ""}>'

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def _copy(self, **kwargs):
        return copy(self)._replace(**kwargs)

    def close(self, portion: float = 1., finalize=False):
        """Place new `Order` to close `portion` of the trade at next market price."""
        assert 0 < portion <= 1, "portion must be a fraction between 0 and 1"
        size = copysign(max(1, round(abs(self.__size) * portion)), -self.__size)
        order = Order(self.__broker, self.__ticker, size, parent_trade=self,
                      entry_time=self.__broker.now, tag=self.__tag)
        if finalize:
            return order
        else:
            self.__broker.orders.insert(0, order)

    # Fields getters

    @property
    def ticker(self):
        return self.__ticker

    @property
    def size(self):
        """Trade size (volume; negative for short trades)."""
        return self.__size

    @property
    def entry_price(self) -> float:
        """Trade entry price."""
        return self.__entry_price

    @property
    def exit_price(self) -> Optional[float]:
        """Trade exit price (or None if the trade is still active)."""
        return self.__exit_price

    @property
    def entry_bar(self) -> int:
        """Candlestick bar index of when the trade was entered."""
        return self.__entry_bar

    @property
    def exit_bar(self) -> Optional[int]:
        """
        Candlestick bar index of when the trade was exited
        (or None if the trade is still active).
        """
        return self.__exit_bar

    @property
    def tag(self):
        """
        A tag value inherited from the `Order` that opened
        this trade.

        This can be used to track trades and apply conditional
        logic / subgroup analysis.

        See also `Order.tag`.
        """
        return self.__tag

    @property
    def _sl_order(self):
        return self.__sl_order

    @property
    def _tp_order(self):
        return self.__tp_order

    # Extra properties

    @property
    def entry_time(self) -> Union[pd.Timestamp, int]:
        """Datetime of when the trade was entered."""
        return self.__broker._data.index[self.__entry_bar]

    @property
    def exit_time(self) -> Optional[Union[pd.Timestamp, int]]:
        """Datetime of when the trade was exited."""
        if self.__exit_bar is None:
            return None
        return self.__broker._data.index[self.__exit_bar]

    @property
    def is_long(self):
        """True if the trade is long (trade size is positive)."""
        return self.__size > 0

    @property
    def is_short(self):
        """True if the trade is short (trade size is negative)."""
        return not self.is_long

    @property
    def pl(self):
        """Trade profit (positive) or loss (negative) in cash units."""
        price = self.__exit_price or self.__broker.last_price(self.__ticker)
        return self.__size * (price - self.__entry_price)

    @property
    def pl_pct(self):
        """Trade profit (positive) or loss (negative) in percent."""
        price = self.__exit_price or self.__broker.last_price(self.__ticker)
        return copysign(1, self.__size) * (price / self.__entry_price - 1) if self.__entry_price != 0 else np.nan

    @property
    def value(self):
        """Trade total value in cash (volume × price)."""
        price = self.__exit_price or self.__broker.last_price(self.__ticker)
        return self.__size * price

    # SL/TP management API

    @property
    def sl(self):
        """
        Stop-loss price at which to close the trade.

        This variable is writable. By assigning it a new price value,
        you create or modify the existing SL order.
        By assigning it `None`, you cancel it.
        """
        return self.__sl_order and self.__sl_order.stop

    @sl.setter
    def sl(self, price: float):
        self.__set_contingent('sl', price)

    @property
    def tp(self):
        """
        Take-profit price at which to close the trade.

        This property is writable. By assigning it a new price value,
        you create or modify the existing TP order.
        By assigning it `None`, you cancel it.
        """
        return self.__tp_order and self.__tp_order.limit

    @tp.setter
    def tp(self, price: float):
        self.__set_contingent('tp', price)

    def __set_contingent(self, type, price):
        assert type in ('sl', 'tp')
        assert price is None or 0 < price < np.inf
        attr = f'_{self.__class__.__qualname__}__{type}_order'
        order: Order = getattr(self, attr)
        if order:
            order.cancel()
        if price:
            kwargs = {'stop': price} if type == 'sl' else {'limit': price}
            order = self.__broker.new_order(self.ticker, -self.size, trade=self, tag=self.tag, **kwargs)
            setattr(self, attr, order)


class _Broker:
    def __init__(self, *, data: _Data, cash, holding, commission, margin, trade_on_close, hedging, exclusive_orders,
                 trade_start_date, lot_size, fail_fast, storage):
        assert 0 < cash, f"cash should be >0, is {cash}"
        assert -.1 <= commission < .1, \
            ("commission should be between -10% "
             f"(e.g. market-maker's rebates) and 10% (fees), is {commission}")
        assert 0 < margin <= 1, f"margin should be between 0 and 1, is {margin}"
        self._data = data
        self._cash = cash
        self._holding = holding
        self._commission = commission
        self._leverage = 1 / margin
        self._trade_on_close = trade_on_close
        self._hedging = hedging
        self._exclusive_orders = exclusive_orders
        self._trade_start_date = trade_start_date   # datetime with no tz
        self._lot_size = lot_size
        self._fail_fast = fail_fast
        self._storage = storage

        self._equity = np.tile(np.nan, (len(data.index), len(data.tickers)+2))
        self.orders: List[Order] = []
        self.trades: Dict[str, List[Trade]] = {ticker: [] for ticker in self._data.tickers}
        self._trade_start_bar = min(
            (self._data.index.tz_localize(None) < self._trade_start_date).sum(),
            len(self._data)-1) if self._trade_start_date else 0
        # Handle preexisting positions as if they are acquired on the first bar but
        # at the close price of trade_start_date, so that the portfolio return is 0
        # between backtest start date and trade_start_date.
        if self._holding:
            for ticker, size in self._holding.items():
                if size:
                    self.trades[ticker].append(Trade(self, ticker=ticker, size=size, entry_price=self._data[
                        ticker, 'Close'][self._trade_start_bar], entry_bar=0, tag='preexisting'))
                    # add the cost for preexisting positions to initial cash
                    self._cash += size * self._data[ticker, 'Close'][self._trade_start_bar]
        self.positions: Dict[str, Position] = {ticker: Position(self, ticker) for ticker in self._data.tickers}
        self.closed_trades: List[Trade] = []

    def __repr__(self):
        pos = ','.join([f'{k}:{p.size}' for k, p in self.positions.items()])
        return f'<Broker: margin_available:{self.margin_available:.0f},{pos} ({len(self.all_trades)} trades)>'

    def rebalance(self, alloc: Allocation, force: bool = False, rtol: float = 0.01, atol: int = 0, cash_reserve: float = 0.1):
        assert 0 <= cash_reserve < 1, "cash_reserve should be between 0 and 1"
        assert 0 <= rtol < 1, "rtol should be between 0 and 1"
        assert 0 <= atol, "atol should be non-negative"

        # ignore any trade actions before trade_start_date
        if self._trade_start_date and self.now.replace(tzinfo=None) < self._trade_start_date:
            alloc._clear()
            return
        # rebalance if force rebalance is true or portfolio weights have changed
        if force or alloc.modified:
            # money value of current portfolio
            total_equity = self.equity()
            # desired values for each ticker excluding cash reserve that is not to be allocated
            value_allocation = alloc.weights * total_equity * (1 - cash_reserve)
            # calculate the amount to buy or sell
            current_value = pd.Series([self.equity(ticker)
                                       for ticker in self._data.tickers], index=self._data.tickers)
            value_diff = value_allocation - current_value
            value_diff_abs = value_diff.abs().sum()
            value_diff_rel = value_diff_abs / total_equity
            # sort in ascending order so that sell orders are placed first then buy orders to make sure that cash
            # balance is always positive in simulation
            for ticker in value_diff.sort_values().index:
                if alloc.weights.loc[ticker] == 0:
                    # this may generate multiple orders for the same ticker if multiple long positions are opened
                    # for the same ticker previously over time
                    for trade in self.trades[ticker]:
                        trade.close()
                else:
                    # rebalance if the current value deviate too much from the desired value
                    # this is to avoid tiny orders triggered by ticker price fluctuation
                    if value_diff[ticker] and (atol and value_diff_abs > atol or value_diff_rel > rtol):
                        # calculate number of shares to buy respecting lot_size
                        # implicitly this forces order in whole share, fractional share not supported for now
                        size = value_diff[ticker] // self.last_price(ticker) // self._lot_size * self._lot_size
                        if size != 0:
                            self.new_order(ticker=ticker, size=size)
        alloc._next()

    def new_order(self,
                  ticker: str,
                  size: float,
                  limit: Optional[float] = None,
                  stop: Optional[float] = None,
                  sl: Optional[float] = None,
                  tp: Optional[float] = None,
                  tag: object = None,
                  *,
                  trade: Optional[Trade] = None):
        """
        Argument size indicates whether the order is long or short
        """
        ticker = ticker or self._data.the_ticker

        # ignore any trade actions before trade_start_date
        if self._trade_start_date and self.now.replace(tzinfo=None) < self._trade_start_date:
            return

        size = float(size)
        stop = stop and float(stop)
        limit = limit and float(limit)
        sl = sl and float(sl)
        tp = tp and float(tp)

        is_long = size > 0
        adjusted_price = self._adjusted_price(ticker, size)

        if is_long:
            if not (sl or -np.inf) < (limit or stop or adjusted_price) < (tp or np.inf):
                raise ValueError(
                    "Long orders require: "
                    f"SL ({sl}) < LIMIT ({limit or stop or adjusted_price}) < TP ({tp})")
        else:
            if not (tp or -np.inf) < (limit or stop or adjusted_price) < (sl or np.inf):
                raise ValueError(
                    "Short orders require: "
                    f"TP ({tp}) < LIMIT ({limit or stop or adjusted_price}) < SL ({sl})")

        order = Order(self, ticker, size, limit, stop, sl, tp, trade, self.now, tag=tag)
        # Put the new order in the order queue,
        # inserting SL/TP/trade-closing orders in-front
        if trade:
            self.orders.insert(0, order)
        else:
            # If exclusive orders (each new order auto-closes previous orders/position),
            # cancel all non-contingent orders and close all open trades beforehand
            if self._exclusive_orders:
                for o in self.orders:
                    if not o.is_contingent:
                        o.cancel()
                for t in self.trades[ticker]:
                    t.close()

            self.orders.append(order)

        return order

    def last_price(self, ticker) -> float:
        """ Price at the last (current) close. """
        return self._data[ticker, 'Close'][-1]

    def _adjusted_price(self, ticker: str, size=None, price=None) -> float:
        """
        Long/short `price`, adjusted for commisions.
        In long positions, the adjusted price is a fraction higher, and vice versa.
        """
        return (price or self.last_price(ticker)) * (1 + copysign(self._commission, size))

    def equity(self, ticker: str = None) -> float:
        if ticker:
            # return current value of the asset
            return sum(trade.value for trade in self.trades[ticker])
        else:
            return self._cash + sum(trade.pl for trade in self.all_trades)

    @property
    def margin_available(self) -> float:
        # From https://github.com/QuantConnect/Lean/pull/3768
        margin_used = sum(abs(trade.value) / self._leverage for trade in self.all_trades)
        return max(0, self.equity() - margin_used)

    @property
    def all_trades(self) -> List[Trade]:
        return [trade for _, trades in self.trades.items() for trade in trades]

    @property
    def now(self):
        return self._data.now

    def finalize(self):
        # Ignore any unprocessed orders in broker.orders since they don't have chance
        # to be executed before the end of backtest. This is not strictly
        # true since market order can still execute if trade_on_close=True.
        # But we ignore this since it won't affect the strategy performance.

        # Close any remaining open trades so they produce some stats
        final_orders = [trade.close(finalize=True) for trade in self.all_trades]
        for order in final_orders:
            price = self.last_price(order.ticker)
            time_index = len(self._data) - 1
            trade = order.parent_trade
            _prev_size = trade.size
            size = copysign(min(abs(_prev_size), abs(order.size)), order.size)
            if trade in self.trades[order.ticker]:
                self._reduce_trade(trade, price, size, time_index)
                assert order.size != -_prev_size or trade not in self.trades[order.ticker]

    def next(self):
        i = len(self._data) - 1
        self._process_orders()

        # Log account equity for the equity curve
        total_equity = self.equity()
        ticker_equity = [self.equity(ticker) for ticker in self._data.tickers]
        equity = [total_equity, *ticker_equity, self.margin_available]
        self._equity[i] = equity

        # If equity is negative, set all to 0 and stop the simulation
        if equity[0] <= 0:
            assert self.margin_available <= 0
            for trade in self.all_trades:
                self._close_trade(trade, self.last_price(trade.ticker), i)
            self._cash = 0
            self._equity[i:] = 0
            raise _OutOfMoneyError

    def _process_orders(self):
        i = len(self._data) - 1
        reprocess_orders = False

        # Process orders
        for order in list(self.orders):  # type: Order

            data = self._data
            open_, high, low = (
                data[order.ticker, 'Open'][-1],
                data[order.ticker, 'High'][-1],
                data[order.ticker, 'Low'][-1])
            prev_close = data[order.ticker, 'Close'][-2]

            # Related SL/TP order was already removed
            if order not in self.orders:
                continue

            # Check if stop condition was hit
            stop_price = order.stop
            if stop_price:
                is_stop_hit = ((high > stop_price) if order.is_long else (low < stop_price))
                if not is_stop_hit:
                    continue

                # > When the stop price is reached, a stop order becomes a market/limit order.
                # https://www.sec.gov/fast-answers/answersstopordhtm.html
                order._replace(stop_price=None)

            # Determine purchase price.
            # Check if limit order can be filled.
            if order.limit:
                is_limit_hit = low < order.limit if order.is_long else high > order.limit
                # When stop and limit are hit within the same bar, we pessimistically
                # assume limit was hit before the stop (i.e. "before it counts")
                is_limit_hit_before_stop = (is_limit_hit and
                                            (order.limit < (stop_price or -np.inf)
                                             if order.is_long
                                             else order.limit > (stop_price or np.inf)))
                if not is_limit_hit or is_limit_hit_before_stop:
                    continue

                # stop_price, if set, was hit within this bar
                price = (min(stop_price or open_, order.limit)
                         if order.is_long else
                         max(stop_price or open_, order.limit))
            else:
                # Market-if-touched / market order
                price = prev_close if self._trade_on_close else open_
                price = (max(price, stop_price or -np.inf)
                         if order.is_long else
                         min(price, stop_price or np.inf))

            # Determine entry/exit bar index
            is_market_order = not order.limit and not stop_price
            time_index = (i - 1) if is_market_order and self._trade_on_close else i

            # If order is a SL/TP order, it should close an existing trade it was contingent upon
            if order.parent_trade:
                trade = order.parent_trade
                _prev_size = trade.size
                # If order.size is "greater" than trade.size, this order is a trade.close()
                # order and part of the trade was already closed beforehand
                size = copysign(min(abs(_prev_size), abs(order.size)), order.size)
                # If this trade isn't already closed (e.g. on multiple `trade.close(.5)` calls)
                if trade in self.trades[order.ticker]:
                    self._reduce_trade(trade, price, size, time_index)
                    assert order.size != -_prev_size or trade not in self.trades[order.ticker]
                if order in (trade._sl_order, trade._tp_order):
                    assert order.size == -trade.size
                    assert order not in self.orders  # Removed when trade was closed
                else:
                    # It's a trade.close() order, now done
                    assert abs(_prev_size) >= abs(size) >= 1
                    self.orders.remove(order)
                continue

            # Else this is a stand-alone trade

            # Adjust price to include commission (or bid-ask spread).
            # In long positions, the adjusted price is a fraction higher, and vice versa.
            adjusted_price = self._adjusted_price(order.ticker, order.size, price)

            # If order size was specified proportionally,
            # precompute true size in units, accounting for margin and spread/commissions
            size = order.size
            if -1 < size < 1:
                size = copysign(int((self.margin_available * self._leverage * abs(size))
                                    // adjusted_price), size)
                # Not enough cash/margin even for a single unit
                if not size:
                    self.orders.remove(order)
                    continue
                else:
                    # replace relative size with calculated size
                    order.size = int(size)
            assert size == round(size)
            need_size = int(size)

            if not self._hedging:
                # Fill position by FIFO closing/reducing existing opposite-facing trades.
                # Existing trades are closed at unadjusted price, because the adjustment
                # was already made when buying.
                for trade in list(self.trades[order.ticker]):
                    if trade.is_long == order.is_long:
                        continue
                    assert trade.size * order.size < 0

                    # Order size greater than this opposite-directed existing trade,
                    # so it will be closed completely
                    if abs(need_size) >= abs(trade.size):
                        self._close_trade(trade, price, time_index)
                        need_size += trade.size
                    else:
                        # The existing trade is larger than the new order,
                        # so it will only be closed partially
                        self._reduce_trade(trade, price, need_size, time_index)
                        need_size = 0

                    if not need_size:
                        break

            # If we don't have enough liquidity to cover for the order, abort the backtest
            if abs(need_size) * adjusted_price > self.margin_available * self._leverage:
                if self._fail_fast:
                    raise RuntimeError(
                        f'Not enough liquidity for {order}, has {int(self.margin_available * self._leverage)},'
                        f' needs {int(abs(need_size) * adjusted_price)}, aborting')
                else:
                    self.orders.remove(order)
                    continue

            # Open a new trade
            if need_size:
                self._open_trade(order.ticker, adjusted_price, need_size, order.sl, order.tp, time_index, order.tag)

                # We need to reprocess the SL/TP orders newly added to the queue.
                # This allows e.g. SL hitting in the same bar the order was open.
                # See https://github.com/kernc/backtesting.py/issues/119
                if order.sl or order.tp:
                    if is_market_order:
                        reprocess_orders = True
                    elif (low <= (order.sl or -np.inf) <= high or
                          low <= (order.tp or -np.inf) <= high):
                        warnings.warn(
                            f"({data.index[-1]}) A contingent SL/TP order would execute in the "
                            "same bar its parent stop/limit order was turned into a trade. "
                            "Since we can't assert the precise intra-candle "
                            "price movement, the affected SL/TP order will instead be executed on "
                            "the next (matching) price/bar, making the result (of this trade) "
                            "somewhat dubious. "
                            "See https://github.com/kernc/backtesting.py/issues/119",
                            UserWarning)

            # Order processed
            self.orders.remove(order)

        if reprocess_orders:
            self._process_orders()

    def _reduce_trade(self, trade: Trade, price: float, size: float, time_index: int):
        assert trade.size * size < 0
        assert abs(trade.size) >= abs(size)

        size_left = trade.size + size
        assert size_left * trade.size >= 0
        if not size_left:
            close_trade = trade
        else:
            # Reduce existing trade ...
            trade._replace(size=size_left)
            if trade._sl_order:
                trade._sl_order._replace(size=-trade.size)
            if trade._tp_order:
                trade._tp_order._replace(size=-trade.size)

            # ... by closing a reduced copy of it
            close_trade = trade._copy(size=-size, sl_order=None, tp_order=None)
            self.trades[trade.ticker].append(close_trade)

        self._close_trade(close_trade, price, time_index)

    def _close_trade(self, trade: Trade, price: float, time_index: int):
        self.trades[trade.ticker].remove(trade)
        if trade._sl_order:
            self.orders.remove(trade._sl_order)
        if trade._tp_order:
            self.orders.remove(trade._tp_order)

        self.closed_trades.append(trade._replace(exit_price=price, exit_bar=time_index))
        self._cash += trade.pl

    def _open_trade(self, ticker: str, price: float, size: int,
                    sl: Optional[float], tp: Optional[float], time_index: int, tag):
        trade = Trade(self, ticker, size, price, time_index, tag)
        self.trades[ticker].append(trade)
        # Create SL/TP (bracket) orders.
        # Make sure SL order is created first so it gets adversarially processed before TP order
        # in case of an ambiguous tie (both hit within a single bar).
        # Note, sl/tp orders are inserted at the front of the list, thus order reversed.
        if tp:
            trade.tp = tp
        if sl:
            trade.sl = sl


class Backtest:
    """
    Backtest a particular (parameterized) strategy
    on particular data.

    Upon initialization, call method
    `minitrade.backtest.core.backtesting.Backtest.run` to run a backtest
    instance, or `minitrade.backtest.core.backtesting.Backtest.optimize` to
    optimize it.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 strategy: Type[Strategy],
                 *,
                 cash: float = 10_000,
                 holding: dict = {},
                 commission: float = .0,
                 margin: float = 1.,
                 trade_on_close=False,
                 hedging=False,
                 exclusive_orders=False,
                 trade_start_date=None,
                 lot_size=1,
                 fail_fast=True,
                 storage: dict | None = None,
                 ):
        """
        Initialize a backtest. Requires data and a strategy to test.

        `data` is a `pd.DataFrame` with 2-level columns:
        1st level is a list of tickers, and 
        2nd level is `Open`, `High`, `Low`, `Close`, and `Volume`.
        If the strategy works only on one asset, the 1st level can be dropped.
        If any columns are missing, set them to what you have available,
        e.g.

            df['Open'] = df['High'] = df['Low'] = df['Close']
            df['Volumn'] = 0

        The passed data frame can contain additional columns that
        can be used by the strategy (e.g. sentiment info).
        DataFrame index can be either a datetime index (timestamps)
        or a monotonic range index (i.e. a sequence of periods).

        `strategy` is a `minitrade.backtest.core.backtesting.Strategy`
        _subclass_ (not an instance).

        `cash` is the initial cash to start with.

        `holding` is a mapping of preexisting assets and their sizes before 
        backtest begins, e.g. 

            {'AAPL': 10, 'MSFT': 5}

        `commission` is the commission ratio. E.g. if your broker's commission
        is 1% of trade value, set commission to `0.01`. Note, if you wish to
        account for bid-ask spread, you can approximate doing so by increasing
        the commission, e.g. set it to `0.0002` for commission-less forex
        trading where the average spread is roughly 0.2‰ of asking price.

        `margin` is the required margin (ratio) of a leveraged account.
        No difference is made between initial and maintenance margins.
        To run the backtest using e.g. 50:1 leverge that your broker allows,
        set margin to `0.02` (1 / leverage).

        If `trade_on_close` is `True`, market orders will be filled
        with respect to the current bar's closing price instead of the
        next bar's open.

        If `hedging` is `True`, allow trades in both directions simultaneously.
        If `False`, the opposite-facing orders first close existing trades in
        a [FIFO] manner.

        If `exclusive_orders` is `True`, each new order auto-closes the previous
        trade/position, making at most a single trade (long or short) in effect
        at each time.

        If `trade_start_date` is not None, orders generated before the date are
        surpressed and ignored in backtesting.

        `lot_size` is the minimum increment of shares you buy in one order. Order 
        size will be rounded to integer multiples during rebalance.

        `fail_fast`, when True, instructs the backtester to bail out when
        cash is not enough to cover an order. This can be used in live trading
        to detect issues early. If False, backtesting will ignore the order and 
        continue, which can be convenient during algorithm research.

        `storage`, when not None, is a dictionary that contains saved states from 
        past runs. Modification to storage is persisted and can be made available 
        for future runs. 

        [FIFO]: https://www.investopedia.com/terms/n/nfa-compliance-rule-2-43b.asp
        """

        if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
            raise TypeError('`strategy` must be a Strategy sub-type')
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas.DataFrame with columns")
        if not isinstance(commission, Number):
            raise TypeError('`commission` must be a float value, percent of '
                            'entry order price')

        data = data.copy(deep=False)
        ohlc = ['Open', 'High', 'Low', 'Close']

        # Convert single asset data into 2-level column index
        if data.columns.nlevels == 1:
            data.columns = pd.MultiIndex.from_product([['Asset'], data.columns])

        # Convert index to datetime index
        if (not isinstance(data.index, pd.DatetimeIndex) and
            not isinstance(data.index, pd.RangeIndex) and
            # Numeric index with most large numbers
            (data.index.is_numeric() and
             (data.index > pd.Timestamp('1975').timestamp()).mean() > .8)):
            try:
                data.index = pd.to_datetime(data.index, infer_datetime_format=True)
            except ValueError:
                pass
        if not set(data.columns.levels[1]).issuperset(set(ohlc)):
            raise ValueError("`data` must be a pandas.DataFrame containing columns 'Open', 'High', 'Low', 'Close'")
        if len(data) == 0:
            raise ValueError("`data` cannot be empty")
        if np.any(data.xs('Close', axis=1, level=1) > cash):
            warnings.warn('Some prices are larger than initial cash value. Note that fractional '
                          'trading is not supported. If you want to trade Bitcoin, '
                          'increase initial cash, or trade μBTC or satoshis instead (GH-134).',
                          stacklevel=2)
        if not data.index.is_monotonic_increasing:
            warnings.warn('Data index is not sorted in ascending order. Sorting.',
                          stacklevel=2)
            data = data.sort_index()
        if data.loc[:, (slice(None), ohlc)].apply(lambda s: s.loc[s.first_valid_index():].isna().sum()).sum() > 0:
            raise ValueError('Some OHLC values are missing (NaN). '
                             'Please strip those lines with `df.dropna()` or '
                             'fill them in with `df.interpolate()` or whatever.')
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.warn('Data index is not datetime. Assuming simple periods, '
                          'but `pd.DateTimeIndex` is advised.',
                          stacklevel=2)
        data.index.name = 'Date'

        self._data = data
        self._broker = partial(
            _Broker, cash=cash, holding=holding, commission=commission, margin=margin,
            trade_on_close=trade_on_close, hedging=hedging,
            exclusive_orders=exclusive_orders,
            trade_start_date=datetime.strptime(trade_start_date, '%Y-%m-%d') if trade_start_date else None,
            lot_size=lot_size, fail_fast=fail_fast, storage=storage,
        )
        self._strategy = strategy
        self._results: Optional[pd.Series] = None

        # equal weighed average, as if buy and hold an equal weighed portfolio
        weights = 1 / self._data.xs('Close', axis=1, level=1).iloc[0]
        weighted_data = self._data.copy()
        weighted_data = weighted_data.loc[:, (slice(None), ohlc)]
        for ticker in weights.index:
            weighted_data[ticker] = weighted_data[ticker] * weights[ticker]
        weighted_data = weighted_data.T.groupby(level=1).agg('sum').T / weights.sum()
        self._ohlc_ref_data = weighted_data

    def run(self, **kwargs) -> pd.Series:
        """
        Run the backtest. Returns `pd.Series` with results and statistics.

        Keyword arguments are interpreted as strategy parameters.

            >>> Backtest(GOOG, SmaCross).run()
            Start                     2004-08-19 00:00:00
            End                       2013-03-01 00:00:00
            Duration                   3116 days 00:00:00
            Exposure Time [%]                     93.9944
            Equity Final [$]                      51959.9
            Equity Peak [$]                       75787.4
            Return [%]                            419.599
            Buy & Hold Return [%]                 703.458
            Return (Ann.) [%]                      21.328
            Volatility (Ann.) [%]                 36.5383
            Sharpe Ratio                         0.583718
            Sortino Ratio                         1.09239
            Calmar Ratio                         0.444518
            Max. Drawdown [%]                    -47.9801
            Avg. Drawdown [%]                    -5.92585
            Max. Drawdown Duration      584 days 00:00:00
            Avg. Drawdown Duration       41 days 00:00:00
            # Trades                                   65
            Win Rate [%]                          46.1538
            Best Trade [%]                         53.596
            Worst Trade [%]                      -18.3989
            Avg. Trade [%]                        2.35371
            Max. Trade Duration         183 days 00:00:00
            Avg. Trade Duration          46 days 00:00:00
            Profit Factor                         2.08802
            Expectancy [%]                        8.79171
            SQN                                  0.916893
            Kelly Criterion                        0.6134
            _strategy                            SmaCross
            _equity_curve                           Eq...
            _trades                       Size  EntryB...
            _orders                              Ticke...
            _positions                           {'GOO...
            _trade_start_bar                           0
            dtype: object

        .. warning::
            You may obtain different results for different strategy parameters.
            E.g. if you use 50- and 200-bar SMA, the trading simulation will
            begin on bar 201. The actual length of delay is equal to the lookback
            period of the `Strategy.I` indicator which lags the most.
            Obviously, this can affect results.
        """
        data = _Data(self._data.copy(deep=False))
        broker: _Broker = self._broker(data=data)
        strategy: Strategy = self._strategy(broker, data, kwargs)
        processed_orders: List[Order] = []
        final_positions = None

        try:
            strategy.init()
        except Exception as e:
            print('Strategy initialization throws exception', e)
            print(traceback.format_exc())
            return

        # Indicators used in Strategy.next()
        indicator_attrs = {attr: indicator for attr, indicator in strategy.__dict__.items()
                           if any([indicator is item for item in strategy._indicators])}

        # Skip first few candles where indicators are still "warming up"
        start = max((indicator.isna().any(axis=1).argmin() if isinstance(indicator, pd.DataFrame)
                     else indicator.isna().argmin() for indicator in indicator_attrs.values()), default=0)
        start = max(start, strategy._start_on_day)

        # Preprocess indicators to numpy array for better performance
        def deframe(df): return df.iloc[:, 0] if isinstance(df, pd.DataFrame) and len(df.columns) == 1 else df
        indicator_attrs_np = {attr: deframe(indicator).to_numpy() for attr, indicator in indicator_attrs.items()}

        # Disable "invalid value encountered in ..." warnings. Comparison
        # np.nan >= 3 is not invalid; it's False.
        with np.errstate(invalid='ignore'):

            for i in range(start, len(self._data)):
                # Prepare data and indicators for `next` call
                data._set_length(i + 1)
                for attr, indicator in indicator_attrs_np.items():
                    setattr(strategy, attr,
                            _Indicator(
                                array=indicator[: i + 1],
                                df=partial(_Indicator.lazy_indexing, indicator_attrs[attr], i + 1)))

                # Handle orders processing and broker stuff
                try:
                    broker.next()
                except _OutOfMoneyError:
                    break

                # Next tick, a moment before bar close
                strategy.next()

                # take note of the orders generated
                processed_orders.extend(broker.orders)
            else:

                # take note of the final positions
                final_positions = ({t: p.size for t, p in broker.positions.items()}
                                   | {'Cash': int(broker.margin_available)})

                if start < len(self._data):
                    broker.finalize()

            # Set data back to full length
            # for future `indicator._opts['data'].index` calls to work
            data._set_length(len(self._data))

            equity = pd.DataFrame(broker._equity, index=data.index,
                                  columns=['Equity', *data.tickers, 'Cash']).bfill().fillna(broker._cash)

            self._results = compute_stats(
                orders=processed_orders,
                trades=broker.closed_trades,
                equity=equity,
                ohlc_data=self._ohlc_ref_data,
                risk_free_rate=0.0,
                strategy_instance=strategy,
                positions=final_positions,
                trade_start_bar=start,
            )

        return self._results.copy()

    def optimize(self, *,
                 maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
                 method: str = 'grid',
                 max_tries: Optional[Union[int, float]] = None,
                 constraint: Optional[Callable[[dict], bool]] = None,
                 return_heatmap: bool = False,
                 return_optimization: bool = False,
                 random_state: Optional[int] = None,
                 **kwargs) -> Union[pd.Series,
                                    Tuple[pd.Series, pd.Series],
                                    Tuple[pd.Series, pd.Series, dict]]:
        """
        Optimize strategy parameters to an optimal combination.
        Returns result `pd.Series` of the best run.

        `maximize` is a string key from the
        `minitrade.backtest.core.backtesting.Backtest.run`-returned results series,
        or a function that accepts this series object and returns a number;
        the higher the better. By default, the method maximizes
        Van Tharp's [System Quality Number](https://google.com/search?q=System+Quality+Number).

        `method` is the optimization method. Currently two methods are supported:

        * `"grid"` which does an exhaustive (or randomized) search over the
          cartesian product of parameter combinations, and
        * `"skopt"` which finds close-to-optimal strategy parameters using
          [model-based optimization], making at most `max_tries` evaluations.

        [model-based optimization]: \
            https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

        `max_tries` is the maximal number of strategy runs to perform.
        If `method="grid"`, this results in randomized grid search.
        If `max_tries` is a floating value between (0, 1], this sets the
        number of runs to approximately that fraction of full grid space.
        Alternatively, if integer, it denotes the absolute maximum number
        of evaluations. If unspecified (default), grid search is exhaustive,
        whereas for `method="skopt"`, `max_tries` is set to 200.

        `constraint` is a function that accepts a dict-like object of
        parameters (with values) and returns `True` when the combination
        is admissible to test with. By default, any parameters combination
        is considered admissible.

        If `return_heatmap` is `True`, besides returning the result
        series, an additional `pd.Series` is returned with a multiindex
        of all admissible parameter combinations, which can be further
        inspected or projected onto 2D to plot a heatmap
        (see `backtesting.lib.plot_heatmaps()`).

        If `return_optimization` is True and `method = 'skopt'`,
        in addition to result series (and maybe heatmap), return raw
        [`scipy.optimize.OptimizeResult`][OptimizeResult] for further
        inspection, e.g. with [scikit-optimize]\
        [plotting tools].

        [OptimizeResult]: \
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        [scikit-optimize]: https://scikit-optimize.github.io
        [plotting tools]: https://scikit-optimize.github.io/stable/modules/plots.html

        If you want reproducible optimization results, set `random_state`
        to a fixed integer random seed.

        Additional keyword arguments represent strategy arguments with
        list-like collections of possible values. For example, the following
        code finds and returns the "best" of the 7 admissible (of the
        9 possible) parameter combinations:

            backtest.optimize(sma1=[5, 10, 15], sma2=[10, 20, 40],
                              constraint=lambda p: p.sma1 < p.sma2)

        .. TODO::
            Improve multiprocessing/parallel execution on Windos with start method 'spawn'.
        """
        if not kwargs:
            raise ValueError('Need some strategy parameters to optimize')

        maximize_key = None
        if isinstance(maximize, str):
            maximize_key = str(maximize)
            stats = self._results if self._results is not None else self.run()
            if maximize not in stats:
                raise ValueError('`maximize`, if str, must match a key in pd.Series '
                                 'result of backtest.run()')

            def maximize(stats: pd.Series, _key=maximize):
                return stats[_key]

        elif not callable(maximize):
            raise TypeError('`maximize` must be str (a field of backtest.run() result '
                            'Series) or a function that accepts result Series '
                            'and returns a number; the higher the better')
        assert callable(maximize), maximize

        have_constraint = bool(constraint)
        if constraint is None:

            def constraint(_):
                return True

        elif not callable(constraint):
            raise TypeError("`constraint` must be a function that accepts a dict "
                            "of strategy parameters and returns a bool whether "
                            "the combination of parameters is admissible or not")
        assert callable(constraint), constraint

        if return_optimization and method != 'skopt':
            raise ValueError("return_optimization=True only valid if method='skopt'")

        def _tuple(x):
            return x if isinstance(x, Sequence) and not isinstance(x, str) else (x,)

        for k, v in kwargs.items():
            if len(_tuple(v)) == 0:
                raise ValueError(f"Optimization variable '{k}' is passed no "
                                 f"optimization values: {k}={v}")

        class AttrDict(dict):
            def __getattr__(self, item):
                return self[item]

        def _grid_size():
            size = int(np.prod([len(_tuple(v)) for v in kwargs.values()]))
            if size < 10_000 and have_constraint:
                size = sum(1 for p in product(*(zip(repeat(k), _tuple(v))
                                                for k, v in kwargs.items()))
                           if constraint(AttrDict(p)))
            return size

        def _optimize_grid() -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
            rand = default_rng(random_state).random
            grid_frac = (1 if max_tries is None else
                         max_tries if 0 < max_tries <= 1 else
                         max_tries / _grid_size())
            param_combos = [dict(params)  # back to dict so it pickles
                            for params in (AttrDict(params)
                                           for params in product(*(zip(repeat(k), _tuple(v))
                                                                   for k, v in kwargs.items())))
                            if constraint(params)  # type: ignore
                            and rand() <= grid_frac]
            if not param_combos:
                raise ValueError('No admissible parameter combinations to test')

            if len(param_combos) > 1000:
                warnings.warn(f'Searching for best of {len(param_combos)} configurations.',
                              stacklevel=2)

            heatmap = pd.Series(np.nan,
                                name=maximize_key,
                                index=pd.MultiIndex.from_tuples(
                                    [p.values() for p in param_combos],
                                    names=next(iter(param_combos)).keys()))

            def _batch(seq):
                n = np.clip(int(len(seq) // (os.cpu_count() or 1)), 1, 300)
                for i in range(0, len(seq), n):
                    yield seq[i:i + n]

            # Save necessary objects into "global" state; pass into concurrent executor
            # (and thus pickle) nothing but two numbers; receive nothing but numbers.
            # With start method "fork", children processes will inherit parent address space
            # in a copy-on-write manner, achieving better performance/RAM benefit.
            backtest_uuid = np.random.random()
            param_batches = list(_batch(param_combos))
            Backtest._mp_backtests[backtest_uuid] = (self, param_batches, maximize)  # type: ignore
            try:
                # If multiprocessing start method is 'fork' (i.e. on POSIX), use
                # a pool of processes to compute results in parallel.
                # Otherwise (i.e. on Windos), sequential computation will be "faster".
                if mp.get_start_method(allow_none=False) == 'fork':
                    with ProcessPoolExecutor() as executor:
                        futures = [executor.submit(Backtest._mp_task, backtest_uuid, i)
                                   for i in range(len(param_batches))]
                        for future in _tqdm(as_completed(futures), total=len(futures),
                                            desc='Backtest.optimize'):
                            batch_index, values = future.result()
                            for value, params in zip(values, param_batches[batch_index]):
                                heatmap[tuple(params.values())] = value
                else:
                    if os.name == 'posix':
                        warnings.warn("For multiprocessing support in `Backtest.optimize()` "
                                      "set multiprocessing start method to 'fork'.")
                    for batch_index in _tqdm(range(len(param_batches))):
                        _, values = Backtest._mp_task(backtest_uuid, batch_index)
                        for value, params in zip(values, param_batches[batch_index]):
                            heatmap[tuple(params.values())] = value
            finally:
                del Backtest._mp_backtests[backtest_uuid]

            best_params = heatmap.idxmax()

            if pd.isnull(best_params):
                # No trade was made in any of the runs. Just make a random
                # run so we get some, if empty, results
                stats = self.run(**param_combos[0])
            else:
                stats = self.run(**dict(zip(heatmap.index.names, best_params)))

            if return_heatmap:
                return stats, heatmap
            return stats

        def _optimize_skopt() -> Union[pd.Series,
                                       Tuple[pd.Series, pd.Series],
                                       Tuple[pd.Series, pd.Series, dict]]:
            try:
                from skopt import forest_minimize
                from skopt.callbacks import DeltaXStopper
                from skopt.learning import ExtraTreesRegressor
                from skopt.space import Categorical, Integer, Real
                from skopt.utils import use_named_args
            except ImportError:
                raise ImportError("Need package 'scikit-optimize' for method='skopt'. "
                                  "pip install scikit-optimize") from None

            nonlocal max_tries
            max_tries = (200 if max_tries is None else
                         max(1, int(max_tries * _grid_size())) if 0 < max_tries <= 1 else
                         max_tries)

            dimensions = []
            for key, values in kwargs.items():
                values = np.asarray(values)
                if values.dtype.kind in 'mM':  # timedelta, datetime64
                    # these dtypes are unsupported in skopt, so convert to raw int
                    # TODO: save dtype and convert back later
                    values = values.astype(int)

                if values.dtype.kind in 'iumM':
                    dimensions.append(Integer(low=values.min(), high=values.max(), name=key))
                elif values.dtype.kind == 'f':
                    dimensions.append(Real(low=values.min(), high=values.max(), name=key))
                else:
                    dimensions.append(Categorical(values.tolist(), name=key, transform='onehot'))

            # Avoid recomputing re-evaluations:
            # "The objective has been evaluated at this point before."
            # https://github.com/scikit-optimize/scikit-optimize/issues/302
            memoized_run = lru_cache()(lambda tup: self.run(**dict(tup)))

            # np.inf/np.nan breaks sklearn, np.finfo(float).max breaks skopt.plots.plot_objective
            INVALID = 1e300
            progress = iter(_tqdm(repeat(None), total=max_tries, desc='Backtest.optimize'))

            @ use_named_args(dimensions=dimensions)
            def objective_function(**params):
                next(progress)
                # Check constraints
                # TODO: Adjust after https://github.com/scikit-optimize/scikit-optimize/pull/971
                if not constraint(AttrDict(params)):
                    return INVALID
                res = memoized_run(tuple(params.items()))
                value = -maximize(res)
                if np.isnan(value):
                    return INVALID
                return value

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', 'The objective has been evaluated at this point before.')

                res = forest_minimize(
                    func=objective_function,
                    dimensions=dimensions,
                    n_calls=max_tries,
                    base_estimator=ExtraTreesRegressor(n_estimators=20, min_samples_leaf=2),
                    acq_func='LCB',
                    kappa=3,
                    n_initial_points=min(max_tries, 20 + 3 * len(kwargs)),
                    initial_point_generator='lhs',  # 'sobel' requires n_initial_points ~ 2**N
                    callback=DeltaXStopper(9e-7),
                    random_state=random_state)

            stats = self.run(**dict(zip(kwargs.keys(), res.x)))
            output = [stats]

            if return_heatmap:
                heatmap = pd.Series(dict(zip(map(tuple, res.x_iters), -res.func_vals)),
                                    name=maximize_key)
                heatmap.index.names = kwargs.keys()
                heatmap = heatmap[heatmap != -INVALID]
                heatmap.sort_index(inplace=True)
                output.append(heatmap)

            if return_optimization:
                valid = res.func_vals != INVALID
                res.x_iters = list(compress(res.x_iters, valid))
                res.func_vals = res.func_vals[valid]
                output.append(res)

            return stats if len(output) == 1 else tuple(output)

        if method == 'grid':
            output = _optimize_grid()
        elif method == 'skopt':
            output = _optimize_skopt()
        else:
            raise ValueError(f"Method should be 'grid' or 'skopt', not {method!r}")
        return output

    @ staticmethod
    def _mp_task(backtest_uuid, batch_index):
        bt, param_batches, maximize_func = Backtest._mp_backtests[backtest_uuid]
        return batch_index, [maximize_func(stats) if stats['# Trades'] else np.nan
                             for stats in (bt.run(**params)
                                           for params in param_batches[batch_index])]

    _mp_backtests: Dict[float, Tuple['Backtest', List, Callable]] = {}

    def plot(self, *, results: pd.Series = None, filename=None, plot_width=None,
             plot_equity=True, plot_return=False, plot_pl=True,
             plot_volume=False, plot_drawdown=False, plot_trades=True,
             smooth_equity=False, relative_equity=True,
             superimpose: Union[bool, str] = False,
             resample=True, reverse_indicators=False,
             show_legend=True, open_browser=True,
             plot_allocation=False, relative_allocation=True,
             plot_indicator=True):
        """
        Plot the progression of the last backtest run.

        If `results` is provided, it should be a particular result
        `pd.Series` such as returned by
        `minitrade.backtest.core.backtesting.Backtest.run` or
        `minitrade.backtest.core.backtesting.Backtest.optimize`, otherwise the last
        run's results are used.

        `filename` is the path to save the interactive HTML plot to.
        By default, a strategy/parameter-dependent file is created in the
        current working directory.

        `plot_width` is the width of the plot in pixels. If None (default),
        the plot is made to span 100% of browser width. The height is
        currently non-adjustable.

        If `plot_equity` is `True`, the resulting plot will contain
        an equity (initial cash plus assets) graph section. This is the same
        as `plot_return` plus initial 100%.

        If `plot_return` is `True`, the resulting plot will contain
        a cumulative return graph section. This is the same
        as `plot_equity` minus initial 100%.

        If `plot_pl` is `True`, the resulting plot will contain
        a profit/loss (P/L) indicator section.

        If `plot_volume` is `True`, the resulting plot will contain
        a trade volume section.

        If `plot_drawdown` is `True`, the resulting plot will contain
        a separate drawdown graph section.

        If `plot_trades` is `True`, the stretches between trade entries
        and trade exits are marked by hash-marked tractor beams.

        If `smooth_equity` is `True`, the equity graph will be
        interpolated between fixed points at trade closing times,
        unaffected by any interim asset volatility.

        If `relative_equity` is `True`, scale and label equity graph axis
        with return percent, not absolute cash-equivalent values.

        If `superimpose` is `True`, superimpose larger-timeframe candlesticks
        over the original candlestick chart. Default downsampling rule is:
        monthly for daily data, daily for hourly data, hourly for minute data,
        and minute for (sub-)second data.
        `superimpose` can also be a valid [Pandas offset string],
        such as `'5T'` or `'5min'`, in which case this frequency will be
        used to superimpose.
        Note, this only works for data with a datetime index.

        If `resample` is `True`, the OHLC data is resampled in a way that
        makes the upper number of candles for Bokeh to plot limited to 10_000.
        This may, in situations of overabundant data,
        improve plot's interactive performance and avoid browser's
        `Javascript Error: Maximum call stack size exceeded` or similar.
        Equity & dropdown curves and individual trades data is,
        `resample` can also be a [Pandas offset string],
        such as `'5T'` or `'5min'`, in which case this frequency will be
        used to resample, overriding above numeric limitation.
        Note, all this only works for data with a datetime index.

        If `reverse_indicators` is `True`, the indicators below the OHLC chart
        are plotted in reverse order of declaration.

        [Pandas offset string]: \
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        If `show_legend` is `True`, the resulting plot graphs will contain
        labeled legends.

        If `open_browser` is `True`, the resulting `filename` will be
        opened in the default web browser.

        If `plot_allocation` is `True`, the resulting plot will contain
        an equity allocation graph section. 

        If `relative_allocation` is `True`, scale and label equity allocation graph axis
        with return percent, not absolute cash-equivalent values.

        If `plot_indicator` is `True`, the resulting plot will contain
        a section for each indicator used in the strategy.
        """
        if results is None:
            if self._results is None:
                raise RuntimeError('First issue `backtest.run()` to obtain results.')
            results = self._results

        indicators = results._strategy._indicators + list(results._strategy._records.values())

        return plot(
            results=results,
            data=self._data,
            baseline=self._ohlc_ref_data,
            indicators=indicators,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            plot_trades=plot_trades,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser,
            plot_allocation=plot_allocation,
            relative_allocation=relative_allocation,
            plot_indicator=plot_indicator)
