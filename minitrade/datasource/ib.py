

import warnings

from minitrade.broker.base import Broker, BrokerAccount
from minitrade.datasource.base import QuoteSource
from minitrade.utils.config import config


class InteractiveBrokersQuoteSource(QuoteSource):
    '''InteractiveBrokers data source'''

    def __init__(self, alias: str | None = None) -> None:
        super().__init__()
        alias = alias or config.sources.ib.account
        if not alias:
            raise ValueError('No IB account specified')
        self.account = BrokerAccount.get_account(alias)
        self.broker = Broker.get_broker(self.account)

    def _format_ticker(self, ticker):
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
