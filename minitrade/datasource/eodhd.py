

import pandas as pd
from eodhd import APIClient

from minitrade.datasource.base import QuoteSource
from minitrade.utils.config import config


class EODHistoricalDataQuoteSource(QuoteSource):
    '''EOD Historical Data source'''

    def __init__(self, api_key: str = None) -> None:
        super().__init__()
        api_key = api_key or config.sources.eodhd.api_key
        if not api_key:
            raise AttributeError('EOD Historical Data API token is not configured')
        self.client = APIClient(api_key=api_key)

    def _format_ticker(self, ticker):
        if '.' in ticker:
            return ticker
        else:
            return f'{ticker}.US'

    def _ticker_timezone(self, ticker):
        # TODO: get timezone from API
        return 'America/New_York'

    def _ticker_calendar(self, ticker):
        # TODO: get market calendar from API
        return 'NYSE'

    def _adjusted_price(self, data):
        df = data.copy()
        ratio = (df['adjusted_close'] / df['close']).to_numpy()
        df['close'] = df['adjusted_close']
        df['open'] = df['open'] * ratio
        df['high'] = df['high'] * ratio
        df['low'] = df['low'] * ratio
        df['volume'] = (df['volume'] / ratio).astype(int)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df.set_index('Date')
        return df

    def _daily_bar(self, ticker, start, end):
        # TODO: get realtime data for today if market is in session
        data = self.client.get_eod_historical_stock_market_data(
            symbol=ticker, period='d', from_date=start, to_date=end, order='a')
        df = self._adjusted_price(pd.DataFrame(data))
        return df

    def _minute_bar(self, ticker: str, start: str, end: str, interval: int):
        raise NotImplementedError()

    def _spot(self, tickers):
        raise NotImplementedError()
