

from twelvedata import TDClient

from minitrade.datasource.base import QuoteSource
from minitrade.utils.config import config


class TwelveDataQuoteSource(QuoteSource):
    '''TwelveDataQuoteSource retrieves data from https://twelvedata.com. (Experimental)
    '''

    def __init__(self, api_key: str = None) -> None:
        super().__init__()
        api_key = api_key or config.sources.twelvedata.api_key
        if not api_key:
            raise AttributeError('TwelveData API key is not configured')
        self.client = TDClient(apikey=api_key)

    def _format_ticker(self, ticker):
        return ticker

    def _ticker_timezone(self, ticker):
        # TODO: get timezone from API
        return 'America/New_York'

    def _ticker_calendar(self, ticker):
        # TODO: get market calendar from API
        return 'NYSE'

    def _daily_bar(self, ticker, start, end):
        df = self.client.time_series(
            symbol=self._format_ticker(ticker),
            interval='1day',
            start_date=start,
            end_date=end,
            outputsize=5000,
            timezone='Exchange',
            order='ASC',
        ).as_pandas()
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                  'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df.index.rename('Date', inplace=True)
        df.index = df.index.tz_localize(None).normalize()
        return df

    def _minute_bar(self, ticker: str, start: str, end: str, interval: int):
        raise NotImplementedError()

    def _spot(self, tickers):
        raise NotImplementedError()
