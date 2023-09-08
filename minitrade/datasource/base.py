from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any
from urllib import request
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
from tqdm.auto import tqdm

from minitrade.utils.mtdb import MTDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'QuoteSource',
    'download_tickers',
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
        elif name == 'IB':
            from .ib import InteractiveBrokersQuoteSource
            return InteractiveBrokersQuoteSource(**kwargs)
        else:
            raise AttributeError(f'Quote source {name} is not supported')

    def today(self, ticker: str) -> datetime:
        '''Get today's date in a ticker's local timezone.'''
        tk = MTDB.get_one('Ticker', 'ticker', ticker)
        dt = datetime.now(ZoneInfo(tk['timezone'])).replace(hour=0, minute=0, second=0, microsecond=0)
        return dt

    def is_trading_now(self, ticker: str) -> bool:
        '''Check if a ticker is trading now.'''
        tk = MTDB.get_one('Ticker', 'ticker', ticker)
        calendar = mcal.get_calendar(tk['calendar'])
        today = datetime.now(ZoneInfo(tk['timezone']))
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
        if isinstance(tickers, str):
            tickers = tickers.split(',')
        return self._spot(tickers)

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
            df = pd.concat(data, axis=1).ffill()
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


def download_tickers():
    import warnings
    warnings.simplefilter("ignore")     # ignore warnings from pandas read_excel

    def download_nasdaq_traded_symbols():
        df = pd.read_csv('ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt',
                         sep='|', skipfooter=2, engine='python')
        df = df[df['Test Issue'] == 'N']
        df.to_sql('NasdaqTraded', MTDB.conn(), if_exists='replace', index=False)

    def download_szse_traded_symbols():
        df = pd.read_excel(
            'http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1110&TABKEY=tab1',
            dtype={'A股代码': str, 'B股代码': str})
        df.to_sql('SzseTraded', MTDB.conn(), if_exists='replace', index=False)

    def download_szse_traded_etfs():
        df = pd.read_excel(
            'http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1105&TABKEY=tab1&random=0.2921736379452743',
            dtype={'基金代码': str})
        df.to_sql('SzseTradedEtf', MTDB.conn(), if_exists='replace', index=False)

    def download_shse_traded_symbols():
        sse_stock_list_url = 'http://query.sse.com.cn//sseQuery/commonExcelDd.do?sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L&type=inParams&CSRC_CODE=&STOCK_CODE=&REG_PROVINCE=&STOCK_TYPE=1&COMPANY_STATUS=2,4,5,7,8'
        request_headers = {
            'X-Requested-With': 'XMLHttpRequest',
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/56.0.2924.87 Safari/537.36', 'Referer': 'http://www.sse.com.cn/assortment/stock/list/share/'}
        req = request.Request(sse_stock_list_url, headers=request_headers)
        result = request.urlopen(req).read()
        df = pd.read_excel(io.BytesIO(result), dtype={'A股代码': str, 'B股代码': str})
        df.to_sql('ShseTraded', MTDB.conn(), if_exists='replace', index=False)

    def download_hkse_traded_symbols():
        df = pd.read_excel(
            'https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx', skiprows=2,
            dtype={'Stock Code': str})
        df.to_sql('HkseTraded', MTDB.conn(), if_exists='replace', index=False)

    steps = [
        (download_nasdaq_traded_symbols, 'select Symbol as ticker, "Security Name" as name from NasdaqTraded', 'NASDAQ', 'America/New_York', ''),
        (download_shse_traded_symbols, 'select A股代码 as ticker, 证券简称 as name from ShseTraded', 'SSE', 'Asia/Shanghai', 'SH'),
        (download_szse_traded_symbols, 'select A股代码 as ticker, A股简称 as name from SzseTraded', 'SSE', 'Asia/Shanghai', 'SZ'),
        (download_szse_traded_etfs, 'select 基金代码 as ticker, 基金简称 as name from SzseTradedEtf', 'SSE', 'Asia/Shanghai', 'SZ'),
        (download_hkse_traded_symbols, 'select "Stock Code" as ticker, "Name of Securities" as name from HkseTraded', 'HKEX', 'Asia/Hongkong', 'HK'),
    ]
    MTDB.conn().execute('drop table if exists Ticker')
    t = tqdm(steps, total=len(steps), unit="step", desc='Downloading stock symbols', leave=False)
    for downloader, sql, calendar, timezone, yahoo_modifier in t:
        downloader()
        df = pd.read_sql(sql, MTDB.conn())
        df['calendar'] = calendar
        df['timezone'] = timezone
        df['yahoo_modifier'] = yahoo_modifier
        df.to_sql('Ticker', MTDB.conn(), if_exists='append', index=False)
    t.close()
