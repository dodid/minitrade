

from datetime import datetime, timedelta

from alpaca.data import (Adjustment, CryptoHistoricalDataClient, Sort,
                         StockBarsRequest, StockHistoricalDataClient,
                         TimeFrame, TimeFrameUnit)

from minitrade.datasource.base import QuoteSource
from minitrade.utils.config import config


class AlpacaQuoteSource(QuoteSource):
    '''AlpacaQuoteSource retrieves data from https://alpaca.markets. (Experimental)
    '''

    def __init__(self, api_key: str = None, api_secret: str = None, use_adjusted: bool = True) -> None:
        super().__init__()
        api_key = api_key or config.sources.alpaca.api_key
        api_secret = api_secret or config.sources.alpaca.api_secret
        self.use_adjusted = use_adjusted
        if not api_key or not api_secret:
            raise AttributeError('Alpaca API key or secret is not configured')
        self.crypto_client = CryptoHistoricalDataClient()
        self.stock_client = StockHistoricalDataClient(api_key, api_secret)

    def _format_ticker(self, ticker):
        return ticker

    def _ticker_timezone(self, ticker):
        # TODO: get timezone from API
        return 'America/New_York'

    def _ticker_calendar(self, ticker):
        # TODO: get market calendar from API
        return 'NYSE'

    def _daily_bar(self, ticker, start, end):
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=datetime.strptime(start, '%Y-%m-%d'),
            end=datetime.strptime(end, '%Y-%m-%d') + timedelta(hours=12) if end else None,
            adjustment=Adjustment.ALL if self.use_adjusted else Adjustment.NONE,
            sort=Sort.ASC)
        df = self.stock_client.get_stock_bars(request_params=request).df.loc[ticker]
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                  'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index.rename('Date', inplace=True)
        df.index = df.index.tz_localize(None).normalize()
        return df

    def _minute_bar(self, ticker: str, start: str, end: str, interval: int):
        raise NotImplementedError()

    def _spot(self, tickers):
        raise NotImplementedError()
