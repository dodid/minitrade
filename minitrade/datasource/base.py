from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

__all__ = [
    'QuoteSource',
]


class QuoteSource(ABC):
    '''
    QuoteSource is a base class that returns quote data for instruments. Extend this class
    to add a concrete implementation to get data from particular data source.
    '''

    AVAILABLE_SOURCES = sorted(['Yahoo', 'EODHistoricalData', 'TwelveData', 'Alpaca',
                               'EastMoney', 'Tiingo', 'InteractiveBrokers', 'CboeIndex', 'CboeFutures'])
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
            ValueError: If the asked data source is not supported
        '''
        if name == 'Yahoo':
            from .yahoo import YahooQuoteSource
            return YahooQuoteSource(**kwargs)
        elif name == 'Tiingo':
            from .tiingo import TiingoQuoteSource
            return TiingoQuoteSource(**kwargs)
        elif name == 'EODHistoricalData':
            from .eodhd import EODHistoricalDataQuoteSource
            return EODHistoricalDataQuoteSource(**kwargs)
        elif name == 'TwelveData':
            from .twelve import TwelveDataQuoteSource
            return TwelveDataQuoteSource(**kwargs)
        elif name == 'Alpaca':
            from .alpaca import AlpacaQuoteSource
            return AlpacaQuoteSource(**kwargs)
        elif name == 'EastMoney':
            from .eastmoney import EastMoneyQuoteSource
            return EastMoneyQuoteSource()
        elif name == 'InteractiveBrokers':
            from .ib import InteractiveBrokersQuoteSource
            return InteractiveBrokersQuoteSource(**kwargs)
        elif name == 'CboeIndex':
            from .cboe_index import CboeIndexQuoteSource
            return CboeIndexQuoteSource(**kwargs)
        elif name == 'CboeFutures':
            from .cboe_futures import CboeFuturesQuoteSource
            return CboeFuturesQuoteSource(**kwargs)
        else:
            raise ValueError(f'Quote source {name} is not supported')

    @abstractmethod
    def _ticker_timezone(self, ticker: str) -> str:
        raise NotImplementedError()

    def ticker_timezone(self, ticker: str) -> str:
        '''Get the timezone of a ticker.'''
        try:
            return self._ticker_timezone(ticker)
        except Exception as e:
            raise RuntimeError(f'Cannot get market timezone for {ticker}') from e

    @abstractmethod
    def _ticker_calendar(self, ticker: str) -> str:
        raise NotImplementedError()

    def ticker_calendar(self, ticker: str) -> str:
        '''Get the calendar name of a ticker.'''
        try:
            return self._ticker_calendar(ticker)
        except Exception as e:
            raise RuntimeError(f'Cannot get market calendar for {ticker}') from e

    def today(self, ticker: str) -> datetime:
        '''Get today's date in a ticker's local timezone.'''
        return datetime.now(ZoneInfo(self.ticker_timezone(ticker))).replace(hour=0, minute=0, second=0, microsecond=0)

    def is_trading_now(self, ticker: str) -> bool:
        '''Check if a ticker is trading now.'''
        calendar = mcal.get_calendar(self.ticker_calendar(ticker))
        today = self.today(ticker)
        schedule = calendar.schedule(
            start_date=(today - timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=(today + timedelta(days=30)).strftime('%Y-%m-%d')
        )
        return calendar.is_open_now(schedule, only_rth=True)

    @abstractmethod
    def _spot(self, tickers: list[str]) -> pd.Series:
        raise NotImplementedError()

    def spot(self, tickers: list[str] | str) -> pd.Series:
        '''Read current quote for a list of `tickers`.

        Args:
            tickers: Tickers as a list of string or a comma separated string without space

        Returns:
            Current quotes indexed by ticker as a pandas Series
        '''
        try:
            if isinstance(tickers, str):
                tickers = tickers.split(',')
            return self._spot(tickers)
        except Exception as e:
            raise RuntimeError(e) from e

    @abstractmethod
    def _daily_bar(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        '''Same as `daily_bar()` for only one ticker.

        This should be overridden in subclass to provide an implemention.

        Returns:
            A dataframe with columns 'Open', 'High', 'Low', 'Close', 'Volume' indexed by datetime
        '''
        raise NotImplementedError()

    def daily_bar(
            self, tickers: list[str] | str,
            start: str = '2000-01-01', end: str = None, align: bool = True, normalize: bool = False,
            num_workers: int = 1) -> pd.DataFrame:
        '''Read end-of-day OHLCV data for a list of `tickers` starting from `start` date and ending on `end` date (both inclusive).

        Args:
            tickers: Tickers as a list of string or a comma separated string without space
            start: Start date in string format 'YYYY-MM-DD'
            end: End date in string format 'YYYY-MM-DD'
            align: True to align data to start on the same date, i.e. drop leading days when not all tickers have data available.
            normalize: True to normalize the close price on the start date to 1 for all tickers and scale all price data accordingly.
            num_workers: Number of parallel workers to use for fetching data.

        Returns:
            A dataframe with 2-level columns, first level being the tickers, and the second level being columns 'Open', 'High', 'Low', 'Close', 'Volume'. The dataframe is indexed by datetime.
        '''
        def fetch_data(ticker):
            return ticker, self._daily_bar(ticker, start, end)

        try:
            if isinstance(tickers, str):
                tickers = tickers.split(',')

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                data = {ticker: None for ticker in tickers}
                futures = [executor.submit(fetch_data, ticker) for ticker in tickers]
                for future in as_completed(futures):
                    ticker, result = future.result()
                    data[ticker] = result

            df = pd.concat(data, axis=1).sort_index().ffill()
            if align:
                start_index = df[df.notna().all(axis=1)].index[0]
                df = df.loc[start_index:, :]
                if normalize:
                    ohlc = ['Open', 'High', 'Low', 'Close']
                    for s in tickers:
                        df.loc[:, (s, ohlc)] = df.loc[:, (s, ohlc)] / df[s].loc[start_index, 'Close']
            elif normalize:
                raise ValueError('normalize=True requires align=True')
            assert isinstance(df.index, pd.DatetimeIndex)
            assert df.index.is_monotonic_increasing
            return df
        except Exception as e:
            raise RuntimeError(e) from e

    def monthly_bar(
            self, tickers: list[str] | str, start: str = '2020-01-01', end: str = None, align: bool = True,
            normalize: bool = False, num_workers: int = 1) -> pd.DataFrame:
        ''' Read monthly OHLCV data for a list of `tickers` starting from `start` date and ending on `end` date (both inclusive).

        Args:
            tickers: Tickers as a list of string or a comma separated string without space
            start: Start date in string format 'YYYY-MM'
            end: End date in string format 'YYYY-MM'
            align: True to align data to start on the same date, i.e. drop leading days when not all tickers have data available.
            normalize: True to normalize the close price on the start date to 1 for all tickers and scale all price data accordingly.
            num_workers: Number of parallel workers to use for fetching data.

        Returns:
            A dataframe with 2-level columns, first level being the tickers, and the second level being columns 'Open', 'High', 'Low', 'Close', 'Volume'. The dataframe is indexed by last day of month.
        '''
        try:
            start = pd.offsets.MonthBegin().rollback(pd.to_datetime(start)).strftime('%Y-%m-%d')
            end = (pd.to_datetime(end) + pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d') if end else None
            daily = self.daily_bar(tickers, start, end, align, normalize, num_workers=num_workers)
            monthly = daily.ta.apply(lambda x: x.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
            }))
            assert isinstance(monthly.index, pd.DatetimeIndex)
            assert monthly.index.is_monotonic_increasing
            return monthly
        except Exception as e:
            raise RuntimeError(e) from e

    @abstractmethod
    def _minute_bar(self, ticker: str, start: str, end: str, interval: int) -> pd.DataFrame:
        '''Same as `minute_bar()` for only one ticker.

        This should be overridden in subclass to provide an implemention.

        Returns:
            A dataframe with columns 'Open', 'High', 'Low', 'Close', 'Volume' indexed by datetime
        '''
        raise NotImplementedError()

    def minute_bar(
            self, tickers: list[str] | str, start: str = None, end: str = None, interval: int = 1,
            num_workers: int = 1) -> pd.DataFrame:
        '''Read minute OHLCV data for a `ticker` starting from `start` date and ending on `end` date (both inclusive).

        Args:
            ticker: Ticker as a string
            start: Start date in string format 'YYYY-MM-DD'
            end: End date in string format 'YYYY-MM-DD'
            interval: Interval in minutes
            num_workers: Number of parallel workers to use for fetching data.

        Returns:
            A dataframe with columns 'Open', 'High', 'Low', 'Close', 'Volume' indexed by datetime
        '''
        def fetch_data(ticker):
            return ticker, self._minute_bar(ticker, start, end, interval)

        try:
            if isinstance(tickers, str):
                tickers = tickers.split(',')

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                data = {ticker: None for ticker in tickers}
                futures = [executor.submit(fetch_data, ticker) for ticker in tickers]
                for future in as_completed(futures):
                    ticker, result = future.result()
                    data[ticker] = result

            df = pd.concat(data, axis=1).sort_index().ffill()
            assert isinstance(df.index, pd.DatetimeIndex)
            assert df.index.is_monotonic_increasing
            return df
        except Exception as e:
            raise RuntimeError(e) from e
