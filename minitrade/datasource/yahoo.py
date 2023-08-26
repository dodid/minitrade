
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from minitrade.datasource import QuoteSource
from minitrade.utils.config import config


class YahooQuoteSource(QuoteSource):

    def __init__(self, proxy: str = None):
        ''' Yahoo data source

        Parameters
        ----------
        proxy: str
            Http proxy URI to override currently setting if not None
        '''
        self.proxy = proxy or config.sources.yahoo.proxy

    def _daily_bar(self, ticker, start, end):
        # Push 1 day out to include "end" in final data
        end_1 = end and (datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        tk = yf.Ticker(ticker)
        df: pd.DataFrame = tk.history(start=start, end=end_1, interval='1d',
                                      auto_adjust=True, proxy=self.proxy, timeout=10)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Today's data from history api are not reliable. Replace them with spot prices.
        today = self.today(ticker)
        if (not df.loc[today:].empty or not end or end >= today.strftime('%Y-%m-%d')) and self.is_trading_now(ticker):
            tz = df.index.tzinfo
            df = df[df.index < today]
            df.index = df.index.tz_convert(timezone.utc)
            row = dict(zip(df.columns, [
                tk.basic_info['open'],
                tk.basic_info['dayHigh'],
                tk.basic_info['dayLow'],
                tk.basic_info['lastPrice'],
                tk.basic_info['lastVolume']]))
            df = pd.concat([df, pd.DataFrame(row, index=[today.astimezone(timezone.utc)])])
            df.index = df.index.tz_convert(tz)
        return df

    def _spot(self, tickers):
        try:
            data = {ticker: yf.Ticker(ticker).basic_info['lastPrice']
                    if self.is_trading_now(ticker) else None for ticker in tickers}
            df = pd.Series(data, name=datetime.now(timezone.utc)).astype(float)
            return df
        except Exception as e:
            raise AttributeError(f'Data error') from e
