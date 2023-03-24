
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

    def _daily_bar(self, ticker, start, end) -> pd.DataFrame:
        df: pd.DataFrame = yf.Ticker(ticker).history(start=start, end=end, interval='1d',
                                                     auto_adjust=True, proxy=self.proxy, timeout=10)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
