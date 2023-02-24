from __future__ import annotations

import logging
import urllib.request
from abc import ABC, abstractmethod

import pandas as pd

from minitrade.utils.mtdb import MTDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'QuoteSource',
    'populate_nasdaq_traded_symbols'
]


class QuoteSource(ABC):

    AVAILABLE_SOURCES = ['Yahoo']

    @staticmethod
    def get_source(name: str, **kwargs) -> QuoteSource:
        ''' Get data source by name

        Parameters
        ----------
        name : str
            Data source name
        kwargs : Dict
            keyword arguments to be passed to the data source

        Returns
        -------
        source : QuoteSource
            The data source

        Raises
        ------
        RuntimeError
            If the asked data source is not supported
        '''
        if name == 'Yahoo':
            from .yahoo import YahooQuoteSource
            return YahooQuoteSource(**kwargs)
        else:
            raise AttributeError(f'Quote source {name} is not supported')

    @abstractmethod
    def _daily_ohlcv(self, ticker: str, start: str, end: str = None) -> pd.DataFrame:
        '''Read end-of-day OHLCV data for `ticker` starting from `start` date and ending on `end` date (both inclusive).

        Parameters
        ----------
        ticker : str
            The stock symbol
        start : str
            Start date in string format 'YYYY-MM-DD'
        end : str
            End date in string format 'YYYY-MM-DD'

        Returns
        -------
        ohlcv
            A dataframe with columns 'Open', 'High', 'Low', 'Close', 'Volume' indexed by datetime 

        Raises
        ------
        RuntimeError
            If getting data fails for any reason
        '''
        raise NotImplementedError()

    def daily_ohlcv(
            self, tickers: list[str] | str,
            start: str = '2020-01-01', end: str = None, align: bool = True, normalize: bool = False) -> pd.DataFrame:
        '''Read end-of-day OHLCV data for a list of tickers starting from `start` date and ending on `end` date (both inclusive).

        Parameters
        ----------
        tickers : list[str] | str
            Tickers as a list of string or as one comma separated string
        start : str
            Start date in string format 'YYYY-MM-DD'
        end : str
            End date in string format 'YYYY-MM-DD'
        align : bool
            True to align data to start on the same date, i.e. drop leading days when not all tickers have quote available.
        normalize : bool
            True to normalize the close price on the start date to 1 for all tickers and scale all price data accordingly.

        Returns
        -------
        ohlcv
            A dataframe with 2-level columns, first level being the tickers, and the second level being columns 'Open', 
            'High', 'Low', 'Close', 'Volume'. The dataframe is indexed by datetime.

        Raises
        ------
        RuntimeError
            If getting data fails for any reason
        '''
        try:
            if isinstance(tickers, str):
                tickers = tickers.split(',')
            data = {ticker: self._daily_ohlcv(ticker, start, end) for ticker in tickers}
            ohlc = ['Open', 'High', 'Low', 'Close']
            for _, df in data.items():
                df.loc[:, ohlc] = df[ohlc].fillna(method='ffill')
                df.loc[:, 'Volume'] = df['Volume'].fillna(0)
            df = pd.concat(data, axis=1)
            if align:
                start_index = df[df.notna().all(axis=1)].index[0]
                df = df.loc[start_index:, :]
                if normalize:
                    for s in tickers:
                        df.loc[:, (s, ohlc)] = df.loc[:, (s, ohlc)] / df[s].loc[start_index, 'Close']
            return df
        except Exception as e:
            raise RuntimeError(f'Reading OHLCV data failed for tickers={tickers}'
                               f' start={start} end={end} align={align} normalize={normalize}') from e


def populate_nasdaq_traded_symbols():
    with urllib.request.urlopen('ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt') as f:
        rows = f.read().decode('utf-8').split('\r\n')
    columns = rows[0].replace(' ', '_').lower().split('|')
    # skip header and footer
    tickers = [dict(zip(columns, row.split('|'))) for row in rows[1:-2]]
    tickers = [ticker for ticker in tickers if ticker['test_issue'] == 'N']
    MTDB.save_objects(tickers, 'nasdaqtraded', on_conflict='update')
