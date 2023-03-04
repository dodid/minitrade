
import pandas as pd
import yfinance as yf

from minitrade.datasource import QuoteSource
from minitrade.utils.config import config


class YahooQuoteSource(QuoteSource):

    def __init__(self, use_proxy: bool = True, proxy: str = None):
        ''' Yahoo data source

        Parameters
        ----------
        use_proxy : bool
            False to not use proxy at all
        proxy: str
            Http proxy URI to override currently setting if not None
        '''
        yc = config.sources.yahoo
        if use_proxy:
            self.proxy = proxy or (yc.proxy if yc.enable_proxy else None)
        else:
            self.proxy = None

    def _daily_bar(self, ticker, start, end) -> pd.DataFrame:
        df: pd.DataFrame = yf.Ticker(ticker).history(start=start, end=end, interval='1d',
                                                     auto_adjust=True, proxy=self.proxy, timeout=10)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
