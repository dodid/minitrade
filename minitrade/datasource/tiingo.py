

import urllib.parse

import pandas as pd

from minitrade.datasource.base import QuoteSource
from minitrade.utils.config import config


class TiingoQuoteSource(QuoteSource):
    '''TiingoQuoteSource retrieves data from https://www.tiingo.com. (Experimental)
    '''

    def __init__(self, api_key: str = None, use_adjusted: bool = True) -> None:
        super().__init__()
        self.api_key = api_key or config.sources.tiingo.api_key
        self.use_adjusted = use_adjusted
        if not self.api_key:
            raise AttributeError('Tiingo API key is not configured')

    def _format_ticker(self, ticker):
        return ticker

    def _ticker_timezone(self, ticker):
        # TODO: get timezone from API
        return 'America/New_York'

    def _ticker_calendar(self, ticker):
        # TODO: get market calendar from API
        return 'NYSE'

    def _daily_bar(self, ticker, start, end):
        url = f'https://api.tiingo.com/tiingo/daily/{urllib.parse.quote_plus(ticker)}/prices?startDate={start}&format=json&resampleFreq=daily&sort=date&token={self.api_key}'
        if end:
            url += f'&endDate={end}'
        df = pd.read_json(url)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        if self.use_adjusted:
            df = df[['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']]
            df.rename(columns={'adjOpen': 'Open', 'adjHigh': 'High', 'adjLow': 'Low',
                               'adjClose': 'Close', 'adjVolume': 'Volume'}, inplace=True)
        else:
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                               'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df.index.rename('Date', inplace=True)
        df.index = df.index.tz_localize(None).normalize()
        return df

    def _minute_bar(self, ticker: str, start: str, end: str, interval: int):
        raise NotImplementedError()

    def _spot(self, tickers):
        raise NotImplementedError()
