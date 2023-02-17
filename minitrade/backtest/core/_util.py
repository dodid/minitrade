import warnings
from numbers import Number
from typing import Dict, List, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd


def try_(lazy_func, default=None, exception=Exception):
    try:
        return lazy_func()
    except exception:
        return default


def _as_str(value) -> str:
    if isinstance(value, (Number, str)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return 'df'
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


class _Data:
    """
    A data array accessor. Provides access by ticker or by OHLCV "columns" or by both. 
    Unlike the original backtesting.py lib, here it returns data as pd.DataFrame or pd.Series instead of _Array.
    """

    def __init__(self, df: pd.DataFrame):
        self.__df = df
        self.__i = len(df)
        self.__pip: Optional[float] = None
        self.__cache: Dict[str, Union[pd.DataFrame, pd.Series]] = {}
        self.__arrays: Dict[str, Union[pd.DataFrame, pd.Series]] = {}
        self.__tickers = list(self.__df.columns.levels[0])
        self._update()

    def __getitem__(self, item):
        return self.__get_array(item)

    def __getattr__(self, item):
        try:
            return self.__get_array(item)
        except KeyError:
            raise AttributeError(f"Column '{item}' not in data") from None

    def _set_length(self, i):
        self.__i = i
        self.__cache.clear()

    def _update(self):
        index = self.__df.index.copy()
        self.__arrays = {
            ticker_col: arr for ticker_col, arr in self.__df.items()} | {
            col: self.__df.xs(col, axis=1, level=1) for col in self.__df.columns.levels[1]} | {
            ticker: self.__df[ticker] for ticker in self.__df.columns.levels[0]}
        # convert single column dataframe into series
        self.__arrays = {key: df.iloc[:, 0] if isinstance(df, pd.DataFrame) and len(
            df.columns) == 1 else df for key, df in self.__arrays.items()}
        # Leave index as Series because pd.Timestamp nicer API to work with
        self.__arrays['__index'] = index

    def __repr__(self):
        i = min(self.__i, len(self.__df)) - 1
        index = self.__arrays['__index'][i]
        items = ', '.join(f'{k}={v}' for k, v in self.__df.iloc[i].items())
        return f'<Data i={i} ({index}) {items}>'

    def __len__(self):
        return self.__i

    @property
    def df(self) -> pd.DataFrame:
        return (self.__df.iloc[:self.__i]
                if self.__i < len(self.__df)
                else self.__df)

    @property
    def pip(self) -> float:
        if self.__pip is None:
            self.__pip = float(10**-np.median([len(s.partition('.')[-1])
                                               for s in self.__arrays['Close'].astype(str)]))
        return self.__pip

    def __get_array(self, key) -> Union[pd.DataFrame, pd.Series]:
        arr = self.__cache.get(key)
        if arr is None:
            arr = self.__cache[key] = self.__arrays[key][:self.__i]
        return arr

    @property
    def Open(self) -> Union[pd.DataFrame, pd.Series]:
        return self.__get_array('Open')

    @property
    def High(self) -> Union[pd.DataFrame, pd.Series]:
        return self.__get_array('High')

    @property
    def Low(self) -> Union[pd.DataFrame, pd.Series]:
        return self.__get_array('Low')

    @property
    def Close(self) -> Union[pd.DataFrame, pd.Series]:
        return self.__get_array('Close')

    @property
    def Volume(self) -> Union[pd.DataFrame, pd.Series]:
        return self.__get_array('Volume')

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.__get_array('__index')

    @property
    def tickers(self) -> List[str]:
        return self.__tickers

    @property
    def the_ticker(self) -> str:
        if len(self.__tickers) == 1:
            return self.__tickers[0]
        else:
            raise ValueError('Ticker must explicitly specified for multi-asset backtesting')

    # Make pickling in Backtest.optimize() work with our catch-all __getattr__
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
