

import warnings

from minitrade.broker.base import Broker, BrokerAccount
from minitrade.datasource.base import QuoteSource
from minitrade.utils.config import config


class InteractiveBrokersQuoteSource(QuoteSource):
    '''InteractiveBrokersQuoteSource retrieves data from InteractiveBrokers and requires a broker account to access its service.

Accepted symbol format:
- To precisely identify a ticker, you should use the IBKR contract ID, e.g. "756733" for SPDR S&P 500 ETF.
- As a convenience, you can also use the ticker symbol, e.g. "SPY" for SPDR S&P 500 ETF. However, the resolution from ticker symbol to contract ID can be ambiguous. If multiple contract IDs are found for the same symbol, the first one, preferrable an U.S. instrument, will be used. For example, "SPY" will be resolved to "756733" (SPY ARCA) instead of "237937002" (SPY ASX). If you want to resolve a ticker symbol to contract ID manually, use the `lookup()` method.

Daily bar:
- The daily OHLCV data returns historical data up to T-1.

Minute bar:
- Not available

Spot price:
- Experimental
    '''

    def __init__(self, alias: str | None = None) -> None:
        super().__init__()
        alias = alias or config.sources.ib.account
        if not alias:
            raise ValueError('No IB account specified')
        self.account = BrokerAccount.get_account(alias)
        self.broker = Broker.get_broker(self.account)

    def _format_ticker(self, ticker):
        if ticker.isnumeric():
            return ticker
        else:
            resolved = self.broker.resolve_tickers(ticker)[ticker]
            if len(resolved) == 0:
                raise ValueError(f'Unknown ticker {ticker}')
            if len(resolved) > 1:
                warnings.warn(f'Ambiguous ticker {ticker} resolved to {resolved[0]["label"]}')
            return resolved[0]['id']

    def _ticker_timezone(self, ticker):
        raise NotImplementedError()

    def _ticker_calendar(self, ticker):
        raise NotImplementedError()

    def _daily_bar(self, ticker, start, end):
        self.broker.connect()
        return self.broker.daily_bar(self._format_ticker(ticker), start, end)

    def _minute_bar(self, ticker: str, start: str, end: str, interval: int):
        raise NotImplementedError()

    def _spot(self, tickers):
        self.broker.connect()
        return self.broker.spot([self._format_ticker(ticker) for ticker in tickers])

    def lookup(self, ticker_css):
        self.broker.connect()
        return self.broker.resolve_tickers(ticker_css)
