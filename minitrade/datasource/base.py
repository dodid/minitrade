from __future__ import annotations

import logging
import urllib.request
from abc import ABC, abstractmethod

import pandas as pd

from minitrade.utils.mtdb import MTDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuoteSource(ABC):

    @staticmethod
    def get_supported_sources() -> list[str]:
        ''' Return supported quote sources '''
        return ['Yahoo']

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
            from .yahoo import QuoteSourceYahoo
            return QuoteSourceYahoo(**kwargs)
        else:
            raise RuntimeError('Quote source {source_name} is not supported')

    @abstractmethod
    def read_daily_ohlcv(self, ticker: str, start: str = '2000-01-01', end: str = None) -> pd.DataFrame:
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

    def read_daily_ohlcv_for_tickers(
            self, tickers: list[str] | str,
            start: str = '2000-01-01', end: str = None, align: bool = True, normalize: bool = False) -> pd.DataFrame:
        '''Read end-of-day OHLCV data for a list of tickers starting from `start` date and ending on `end` date (both inclusive).

        Parameters
        ----------
        ticker_space : list[str] | str
            A list of tickers or tickers in comma separated string format
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
            ddir = {s: self.read_daily_ohlcv(s, start, end) for s in tickers}
            df = pd.concat(ddir, axis=1)
            ohlc = ['Open', 'High', 'Low', 'Close']
            df.loc[:, (slice(None), 'Volume')] = df.loc[:, (slice(None), 'Volume')].fillna(0)
            df.loc[:, (slice(None), ohlc)] = df.loc[:, (slice(None), ohlc)].fillna(method='ffill')
            if align:
                start_index = df[df.notna().all(axis=1)].index[0]
                df = df.loc[start_index:, :]
                if normalize:
                    for s in tickers:
                        df.loc[:, (s, ohlc)] = df.loc[:, (s, ohlc)] / df[s].loc[start_index, 'Close']
            return df
        except Exception as e:
            raise RuntimeError(
                f'Reading OHLCV data failed for tickers={tickers} start={start} end={end} align={align} normalize={normalize}') from e


class SymbolSource:
    @staticmethod
    def nasdaq_traded():
        with urllib.request.urlopen('ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt') as f:
            rows = f.read().decode('utf-8').split('\r\n')
        columns = rows[0].replace(' ', '_').lower().split('|')
        # skip header and footer
        tickers = [dict(zip(columns, row.split('|'))) for row in rows[1:-2]]
        tickers = [ticker for ticker in tickers if ticker['test_issue'] == 'N']
        MTDB.save_objects(tickers, 'nasdaqtraded', on_conflict='update')
