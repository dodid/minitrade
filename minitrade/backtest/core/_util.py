import warnings
from datetime import datetime
from functools import partial
from numbers import Number
from typing import Callable, Dict, List, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
from pandas_ta import AnalysisIndicators


def try_(lazy_func, default=None, exception=Exception):
    try:
        return lazy_func()
    except exception:
        return default


def _as_str(value) -> str:
    if isinstance(value, (Number, str)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.attrs.get('name', None) or 'df'
    name = str(getattr(value, 'name', '') or '')
    if name in ('Open', 'High', 'Low', 'Close', 'Volume'):
        return name[:1]
    if callable(value):
        name = getattr(value, '__name__', value.__class__.__name__).replace('<lambda>', 'λ')
    if len(name) > 10:
        name = name[:9] + '…'
    return name


def _as_list(value) -> List:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value]


def _data_period(index) -> Union[pd.Timedelta, Number]:
    """Return data index period as pd.Timedelta"""
    values = pd.Series(index[-100:])
    return values.diff().dropna().median()


class _Array(np.ndarray):
    """Array with a corresponding DataFrame/Series attachment."""

    def __new__(cls, array, df: Union[Callable, pd.DataFrame, pd.Series]):
        if not callable(df) and not isinstance(df, (pd.DataFrame, pd.Series)):
            raise ValueError(f'df must be callable or pd.DataFrame/pd.Series. Got {type(df)}')
        obj = np.asarray(array).view(cls)
        obj.__df = df
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.__df = getattr(obj, '__df', None)

    # Make sure __df are carried over when (un-)pickling.
    def __reduce__(self):
        value = super().__reduce__()
        return value[:2] + (value[2] + (self.__dict__,),)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])
        super().__setstate__(state[:-1])

    def __repr__(self) -> str:
        return super().__repr__() + f'\nwith df:\n{self.__df}'

    @property
    def df(self) -> Union[pd.DataFrame, pd.Series]:
        if callable(self.__df):
            self.__df = self.__df()
        return self.__df

    @property
    def s(self) -> pd.Series:
        if isinstance(self.df, pd.Series):
            return self.df
        else:
            raise ValueError(f'Value is not a pd.Series. Shape {self.df.shape}')

    @staticmethod
    def lazy_indexing(df, idx):
        return df.iloc[:idx]


class _Indicator(_Array):
    pass


class _Data:
    """
    A data array accessor. Provides access to OHLCV "columns"
    as a standard `pd.DataFrame` would, except it's not a DataFrame
    and the returned "series" are _not_ `pd.Series` but `np.ndarray`
    for performance reasons.
    """

    def __init__(self, df: pd.DataFrame):
        self.__df = df
        self.__i = len(df)
        self.__pip: Optional[float] = None
        self.__cache: Dict[str, np.ndarray] = {}
        self.__arrays: Dict[str, np.ndarray] = {}
        self.__tickers = list(self.__df.columns.levels[0])
        self.__ta = _TA(self.__df)
        self._update()

    def __getitem__(self, item):
        if item == slice(None, None, None):
            item = None
        return self.__get_array(item)

    def __getattr__(self, item):
        try:
            return self.__get_array(item)
        except KeyError as e:
            raise AttributeError(f"Column '{item}' not in data") from e

    def _set_length(self, i):
        self.__i = i
        self.__cache.clear()

    def _update(self):
        # cache slices of the data as DataFrame/Series for faster access
        arrays = (
            {ticker_col: arr for ticker_col, arr in self.__df.items()}
            | {col: self.__df.xs(col, axis=1, level=1) for col in self.__df.columns.levels[1]}
            | {ticker: self.__df[ticker] for ticker in self.__df.columns.levels[0]}
            | {None: self.__df[self.the_ticker] if len(self.__tickers) == 1 else self.__df}
            | {'__index': self.__df.index.copy()}
        )
        arrays = {key: df.iloc[:, 0] if isinstance(df, pd.DataFrame) and len(
            df.columns) == 1 else df for key, df in arrays.items()}
        # keep another copy as Numpy array
        self.__arrays = {key: (df.to_numpy(), df) for key, df in arrays.items()}

    def __repr__(self):
        i = min(self.__i, len(self.__df)) - 1
        index = self.__arrays['__index'][0][i]
        items = ', '.join(f'{k}={v}' for k, v in self.__df.iloc[i].items())
        return f'<Data i={i} ({index}) {items}>'

    def __len__(self):
        return self.__i

    @property
    def df(self) -> pd.DataFrame:
        df_ = self.__df[self.the_ticker] if len(self.tickers) == 1 else self.__df
        return df_.iloc[:self.__i] if self.__i < len(df_) else df_

    @property
    def pip(self) -> float:
        if self.__pip is None:
            self.__pip = float(10**-np.median([len(s.partition('.')[-1])
                                               for s in self.__arrays['Close'][0].ravel().astype(str)]))
        return self.__pip

    def __get_array(self, key) -> _Array:
        arr = self.__cache.get(key)
        if arr is None:
            array, df = self.__arrays[key]
            if key == '__index':
                arr = self.__cache[key] = _Indicator(array=array[:self.__i], df=lambda: df[:self.__i])
            else:
                arr = self.__cache[key] = _Indicator(array=array[:self.__i], df=lambda: df.iloc[:self.__i])
        return arr

    @property
    def Open(self) -> _Array:
        return self.__get_array('Open')

    @property
    def High(self) -> _Array:
        return self.__get_array('High')

    @property
    def Low(self) -> _Array:
        return self.__get_array('Low')

    @property
    def Close(self) -> _Array:
        return self.__get_array('Close')

    @property
    def Volume(self) -> _Array:
        return self.__get_array('Volume')

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.__get_array('__index').df   # return pd.DatetimeIndex

    # Make pickling in Backtest.optimize() work with our catch-all __getattr__
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    @property
    def now(self) -> datetime:
        return self.index[-1]

    @property
    def tickers(self) -> List[str]:
        return self.__tickers

    @property
    def the_ticker(self) -> str:
        if len(self.__tickers) == 1:
            return self.__tickers[0]
        else:
            raise ValueError('Ticker must explicitly specified for multi-asset backtesting')

    @property
    def ta(self) -> '_TA':
        return self.__ta


try:
    # delete the accessor created by pandas_ta to avoid warning
    del pd.DataFrame.ta
except AttributeError:
    pass


class PicklableAnalysisIndicators(AnalysisIndicators):
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


@pd.api.extensions.register_dataframe_accessor("ta")
class _TA:
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(self, df: pd.DataFrame):
        if df.empty:
            return
        self.__df = df
        if self.__df.columns.nlevels == 2:
            self.__tickers = list(self.__df.columns.levels[0])
            self.__indicators = {ticker: PicklableAnalysisIndicators(df[ticker]) for ticker in self.__tickers}
        elif self.__df.columns.nlevels == 1:
            self.__tickers = []
            self.__indicator = PicklableAnalysisIndicators(df)
        else:
            raise AttributeError(
                f'df.columns can have at most 2 levels, got {self.__df.columns.nlevels}')

    def __ta(self, method, *args, columns=None, **kwargs):
        if self.__tickers:
            dir_ = {ticker: getattr(indicator, method)(*args, **kwargs)
                    for ticker, indicator in self.__indicators.items()}
            if columns:
                for _, df in dir_.items():
                    df.columns = columns
            return pd.concat(dir_, axis=1) if len(dir_) > 1 else dir_[self.__tickers[0]]
        else:
            return getattr(self.__indicator, method)(*args, **kwargs)

    def __getattr__(self, method: str):
        return partial(self.__ta, method)

    def apply(self, func, *args, **kwargs):
        if self.__tickers:
            dir_ = {ticker: func(self.__df[ticker], *args, **kwargs) for ticker in self.__tickers}
            return pd.concat(dir_, axis=1)
        else:
            return func(self.__df, *args, **kwargs)

    def join(self, df, lsuffix='', rsuffix=''):
        if self.__tickers:
            dir_ = {ticker: self.__df[ticker].join(df[ticker], lsuffix=lsuffix, rsuffix=rsuffix)
                    for ticker in self.__tickers}
            return pd.concat(dir_, axis=1)
        else:
            return self.__df.join(df, lsuffix=lsuffix, rsuffix=rsuffix)
