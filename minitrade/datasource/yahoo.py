
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from minitrade.datasource import QuoteSource
from minitrade.utils.config import config


class YahooQuoteSource(QuoteSource):
    '''YahooQuoteSource uses `yfinance` to access the financial data available on Yahoo Finance. 

Yahoo Finance offers an excellent range of market data on stocks, bonds, currencies and cryptocurrencies.

Usage:
- Returned data are adjusted for splits and dividends by default. 
- Pass `use_adjusted=False` to `get_source()` to get unadjusted data.

Accepted symbol format:
- U.S. stocks: AAPL, MSFT, BRK-B, ...
- U.S. ETF: SPY, QQQ, ...
- U.S. index: ^GSPC, ^DJI, ^IXIC, ^RUT, ^VIX, ...
- U.S. futures: ES=F, YM=F, NQ=F, RTY=F, CL=F, GC=F, SI=F, HG=F, ...
- U.S. options: TSLA260116C00210000, TSLA260116P00210000, ...
- U.S. mutual fund: VWUSX, ...
- U.S. bond: ^TNX, ^TYX, ^FVX, ^IRX, ...
- U.S. currency: JPY=X, EURUSD=X, GBPUSD=X, ...
- U.S. commodity: CL=F, GC=F, SI=F, HG=F, ...
- Worldwide stocks: 0700.HK, 000001.SS, BMW.DE, VOD.L, ...
- Worldwide ETF: 2800.HK, 510050.SS, ...
- Worldwide index: ^HSI, ^N225, ^FTSE, ...
- Crypto: BTC-USD, ETH-USD, ...

Daily bar:
- The daily OHLCV data includes historical data up to T-1 and the current spot price at T if it is available. Please note that the spot price might not be available for all tickers, could be delayed, and may not be entirely accurate. Therefore, it is advisable to verify the data quality specifically for the tickers you are interested in before use.

Minute bar:
- Experimental

Spot price:
- Experimental
    '''

    def __init__(self, proxy: str = None, use_adjusted=True):
        ''' Yahoo data source

        Parameters
        ----------
        proxy: str
            Http proxy URI to override currently setting if not None
        '''
        self.proxy = proxy or config.sources.yahoo.proxy
        self.use_adjusted = use_adjusted

    def _format_ticker(self, ticker):
        return ticker

    def _ticker_timezone(self, ticker):
        return yf.Ticker(self._format_ticker(ticker)).fast_info['timezone']

    def _ticker_calendar(self, ticker):
        mapping = {
            'WCB': 'NYSE',   # Chicago Board of Options Exchange (CBOE)
            'NMS': 'NYSE',   # National Market System
            'NGM': 'NYSE',   # Nasdaq global market
            'NYQ': 'NYSE',   # NYSE
            'PCX': 'NYSE',   # Pacific Stock Exchange, NYSE Arca
            'CXI': 'NYSE',   # ^VIX
            'PNK': 'NYSE',   # Pink Sheets OTC Market
            'SSH': 'SSE',    # Shanghai stock exchange
            'SHZ': 'SSE',    # Shenzhen stock exchange
            'HKG': 'HKEX',   # Hong Kong stock exchange
            'BTS': 'NYSE',   # BATS Global Markets
        }
        exch = yf.Ticker(self._format_ticker(ticker)).fast_info['exchange']
        if exch in mapping:
            return mapping[exch]
        else:
            warnings.warn(f'Unknown exchange {exch} for {ticker}')
            return 'NYSE'

    def _daily_bar(self, ticker, start, end):
        # Push 1 day out to include "end" in final data
        end_1 = end and (datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        # Yahoo finance uses '-' instead of '.' in ticker symbol
        tk = yf.Ticker(self._format_ticker(ticker))
        df: pd.DataFrame = tk.history(start=start, end=end_1, interval='1d',
                                      auto_adjust=self.use_adjusted, proxy=self.proxy, timeout=10)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Today's data from history api are not reliable. Replace them with spot prices.
        today = self.today(ticker)
        if (not df.loc[today:].empty or not end or end >= today.strftime('%Y-%m-%d')) and self.is_trading_now(ticker):
            tz = df.index.tzinfo
            df = df[df.index < today]
            df.index = df.index.tz_convert(timezone.utc)
            row = dict(zip(df.columns, [
                tk.fast_info['open'],
                tk.fast_info['dayHigh'],
                tk.fast_info['dayLow'],
                tk.fast_info['lastPrice'],
                tk.fast_info['lastVolume']]))
            df = pd.concat([df, pd.DataFrame(row, index=[today.astimezone(timezone.utc)])])
            df.index = df.index.tz_convert(tz)
        # drop timezone info and keep date only
        df.index.rename('Date', inplace=True)
        df.index = df.index.tz_localize(None).normalize()
        return df

    def _minute_bar(self, ticker: str, start: str, end: str, interval: int):
        # Supported intervals: 1, 2, 5, 15, 30, 60
        if interval not in [1, 2, 5, 15, 30, 60]:
            raise ValueError(f'Interval {interval} is not supported')
        # resolution vs range per request, 1m - 7d, 2m - 60d, 5m - 60d, 15m - 60d, 30m - 60d, 60m - 730d
        period = {1: 7, 2: 60, 5: 60, 15: 60, 30: 60, 60: 730}
        # Push 1 day out to include "end" in final data
        end_1 = end and (datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        tk = yf.Ticker(self._format_ticker(ticker))
        if start:
            df: pd.DataFrame = tk.history(start=start, end=end_1, interval=f'{interval}m',
                                          auto_adjust=True, proxy=self.proxy, timeout=10)
        else:
            df: pd.DataFrame = tk.history(period=f'{period[interval]}d', interval=f'{interval}m',
                                          auto_adjust=True, proxy=self.proxy, timeout=10)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index = df.index.tz_convert(timezone.utc)
        return df

    def _spot(self, tickers):
        try:
            data = {ticker: yf.Ticker(self._format_ticker(ticker)).fast_info['lastPrice']
                    if self.is_trading_now(ticker) else None for ticker in tickers}
            df = pd.Series(data, name=datetime.now(timezone.utc)).astype(float)
            return df
        except Exception as e:
            raise AttributeError(f'Data error') from e
