from __future__ import annotations

import logging
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from minitrade.utils.mtdb import MTDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'QuoteSource',
    'populate_nasdaq_traded_symbols'
]


class QuoteSource(ABC):
    '''
    QuoteSource is a base class that returns quote data for instruments. Extend this class 
    to add a concrete implementation to get data from particular data source.
    '''

    AVAILABLE_SOURCES = ['Yahoo', 'EastMoney']
    '''A list of names for supported quote sources as input to `QuoteSource.get_source()`.'''

    @staticmethod
    def get_source(name: str, **kwargs: dict[str, Any]) -> QuoteSource:
        '''Get quote source by name

        Args:
            name: Quote source name
            kwargs: Keyword arguments to be passed to the quote source constructor

        Returns:
            A QuoteSource instance

        Raises:
            AttributeError: If the asked data source is not supported
        '''
        if name == 'Yahoo':
            from .yahoo import YahooQuoteSource
            return YahooQuoteSource(**kwargs)
        elif name == 'EastMoney':
            from .eastmoney import EastMoneyQuoteSource
            return EastMoneyQuoteSource()
        else:
            raise AttributeError(f'Quote source {name} is not supported')

    @abstractmethod
    def _daily_bar(self, ticker: str, start: str, end: str = None) -> pd.DataFrame:
        '''Same as `daily_bar()` for only one ticker.

        This should be overridden in subclass to provide an implemention.

        Returns:
            A dataframe with columns 'Open', 'High', 'Low', 'Close', 'Volume' indexed by datetime 
        '''
        raise NotImplementedError()

    def daily_bar(
            self, tickers: list[str] | str,
            start: str = '2020-01-01', end: str = None, align: bool = True, normalize: bool = False) -> pd.DataFrame:
        '''Read end-of-day OHLCV data for a list of `tickers` starting from `start` date and ending on `end` date (both inclusive).

        Args:
            tickers: Tickers as a list of string or a comma separated string without space
            start: Start date in string format 'YYYY-MM-DD'
            end: End date in string format 'YYYY-MM-DD'
            align: True to align data to start on the same date, i.e. drop leading days when not all tickers have data available.
            normalize: True to normalize the close price on the start date to 1 for all tickers and scale all price data accordingly.

        Returns:
            A dataframe with 2-level columns, first level being the tickers, and the second level being columns 'Open', 'High', 'Low', 'Close', 'Volume'. The dataframe is indexed by datetime.
        '''
        try:
            if isinstance(tickers, str):
                tickers = tickers.split(',')
            data = {ticker: self._daily_bar(ticker, start, end) for ticker in tickers}
            df = pd.concat(data, axis=1).fillna(method='ffill')
            if align:
                start_index = df[df.notna().all(axis=1)].index[0]
                df = df.loc[start_index:, :]
                if normalize:
                    ohlc = ['Open', 'High', 'Low', 'Close']
                    for s in tickers:
                        df.loc[:, (s, ohlc)] = df.loc[:, (s, ohlc)] / df[s].loc[start_index, 'Close']
            return df
        except Exception:
            raise AttributeError(f'Data error')


def populate_nasdaq_traded_symbols():
    with urllib.request.urlopen('ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt') as f:
        rows = f.read().decode('utf-8').split('\r\n')
    columns = rows[0].replace(' ', '_').lower().split('|')
    # skip header and footer
    tickers = [dict(zip(columns, row.split('|'))) for row in rows[1:-2]]
    tickers = [ticker for ticker in tickers if ticker['test_issue'] == 'N']
    MTDB.save(tickers, 'NasdaqTraded', on_conflict='update')
