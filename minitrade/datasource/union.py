from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from minitrade.datasource.base import QuoteSource
from minitrade.utils.mtdb import MTDB


class UnionQuoteSource(QuoteSource):
    '''UnionQuoteSource allows retrieving data from several underlying data sources. One can configure, for each ticker, the data source and parameters to use for retrieving data.

The data source is configured in the following format:

    [(ticker, source, params), ...]

Where:
- ticker: the ticker or comma separated list of tickers to retrieve data for.
- source: the data source name as accepted by the `QuoteSource.get_source()`.
- params: the additonal parameters to pass to `QuoteSource.get_source()`.

Example:
```
data_source = QuoteSource.get_source('Union', config = [
    ('756733', 'IB', {'alias': 'my_ib_account}),                        # Retrieve SPY from Interactive Brokers ('756733' is the contract id for SPY)
    ('VX', 'CboeFutures', {'cf_method': 'backward', 'roll_day': -7}),   # Retrieve VIX futures from Cboe Futures
    ('AAPL,MSFT', 'Yahoo', None),                                       # Retrieve AAPL and MSFT from Yahoo
    (None, 'Yahoo', None)                                               # Retrieve all other tickers from Yahoo
])
```

Accepted symbol format:
- Same as the underlying data source.

Daily bar:
- Data recency is determined by the most recent data source. For example, if source A provides data up to yesterday (T-1) and source B provides data up to today (T), the combined data from sources A and B will be up to today (T), while A's data at time T is forward filled by data at T-1.

Minute bar:
- Not available

Spot price:
- Not available
    '''

    def __init__(self, config: list = []) -> None:
        super().__init__()
        self.source_map = {}
        for tickers, name, params in config:
            source = QuoteSource.get_source(name, **(params or {}))
            if tickers:
                self.source_map |= {ticker: source for ticker in tickers.split(',')}
            elif None not in self.source_map:
                self.source_map[None] = source
            else:
                raise ValueError('Only one fallback data source is allowed.')

    def _format_ticker(self, ticker):
        return ticker

    def _ticker_timezone(self, ticker):
        if ticker in self.source_map:
            return self.source_map[ticker].ticker_timezone(ticker)
        elif None in self.source_map:
            return self.source_map[None].ticker_timezone(ticker)
        else:
            raise ValueError(f'Ticker {ticker} is not found.')

    def _ticker_calendar(self, ticker):
        if ticker in self.source_map:
            return self.source_map[ticker].ticker_calendar(ticker)
        elif None in self.source_map:
            return self.source_map[None].ticker_calendar(ticker)
        else:
            raise ValueError(f'Ticker {ticker} is not found.')

    def _daily_bar(self, ticker, start, end):
        if ticker in self.source_map:
            return self.source_map[ticker].daily_bar(ticker, start, end)[ticker]
        elif None in self.source_map:
            return self.source_map[None].daily_bar(ticker, start, end)[ticker]
        else:
            raise ValueError(f'Ticker {ticker} is not found.')

    def _minute_bar(self, ticker, start, end, interval):
        raise NotImplementedError()

    def _spot(self, tickers):
        raise NotImplementedError()


@dataclass(kw_only=True)
class UnionQuoteSourceConfig:

    name: str
    '''Data source name'''

    config: list
    '''Configuration in the format of [(ticker, source, params), ...]'''

    update_time: datetime
    '''Time when the union source is updated'''

    @staticmethod
    def list() -> list[UnionQuoteSourceConfig]:
        '''Return the list of available union source configs

        Returns:
            A list of zero or more union source configs
        '''
        configs = MTDB.get_all('UnionQuoteSourceConfig', orderby='name', cls=UnionQuoteSourceConfig)
        return [x.name for x in configs if x.name not in QuoteSource.SYSTEM_SOURCES]

    @staticmethod
    def get_config(name: str) -> UnionQuoteSourceConfig:
        '''Look up a union source config by name

        Args:
            name: Union name

        Returns:
            Union source config if found or None
        '''
        return MTDB.get_one('UnionQuoteSourceConfig', 'name', name, cls=UnionQuoteSourceConfig)

    def save(self) -> None:
        '''Save or update a union source config to database.
        '''
        MTDB.save('UnionQuoteSourceConfig', self, on_conflict='update')

    def delete(self) -> None:
        '''Delete a union source config from database.
        '''
        MTDB.delete('UnionQuoteSourceConfig', 'name', self.name)
