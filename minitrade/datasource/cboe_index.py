import json
from io import StringIO

import pandas as pd
import requests

from minitrade.datasource.base import QuoteSource


class CboeIndexQuoteSource(QuoteSource):
    '''CBOEIndexQuoteSource retrieves historical index data from CBOE website as listed on https://www.cboe.com/indices/.

Accepted symbol format:
- CBOE index: ^VIX, ^CALD, ^PUTD, ...

Daily bar:
- The daily OHLCV data returns historical data up to T-1.

Minute bar:
- Not available

Spot price:
- Not available
    '''

    url = 'https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/{}.json'

    def _format_ticker(self, ticker):
        # Convert index ticker in Yahoo Finance style to CBOE style
        return ticker.replace('^', '_')

    def _ticker_timezone(self, ticker):
        return 'America/New_York'

    def _ticker_calendar(self, ticker):
        return 'NYSE'

    def _daily_bar(self, ticker, start, end):
        url = self.url.format(self._format_ticker(ticker))
        data = requests.get(url).json()
        df = pd.read_json(StringIO(json.dumps(data['data'])))
        df = df[df['date'] >= start]
        if end:
            df = df[df['date'] <= end]
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index.rename('Date', inplace=True)
        df.index = df.index.tz_localize(None).normalize()
        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low',
                  'open': 'Open', 'volume': 'Volume'}, inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df

    def _minute_bar(self, ticker, start, end, interval):
        raise NotImplementedError()

    def _spot(self, tickers):
        raise NotImplementedError()
