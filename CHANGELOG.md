# Minitrade Changelog

## 0.1.35
- [Fix] Fix QuoteSource may return data not in ascending time order in some cases.

## 0.1.34
- [New] QuoteSource supports getting data in parallel when `num_workers` > 1.
- [New] Add performance tearsheet for IB account.
- [New] Add CBOE futures data source.
- [Fix] Change equal weight buy-n-hold benchmark to start on trading start date.

## 0.1.33
- [Fix] Fix process status display on `Home` page.

## 0.1.32
- [New] Add `monthly_bar()` to QuoteSource, which allow for backtesting using monthly data for strategy development. (Trading is still on daily bar.)
- [New] Add database statistics to `Home` page.
- [New] Add CBOE index data source.
- [New] Add FRED data explorer.
- [New] Add tasks in task repo at https://github.com/dodid/minitrade/tree/main/tasks. One can pick and choose.
- [New] Add `About` tab to data source page for usage and documentation.
- [Change] Remove ticker converstion from e.g. "BRK.B" to "BRK-B" in YahooQuoteSource. Use dash in cases like "BRK-B".

## 0.1.31
- [New] Allow scheduling arbitrary tasks
- [New] Add system status monitoring to `Home` page.
- [New] Add experimental data sources: IB, Tiingo, TwelveData, Alpaca, EODHD.
- [New] Inspect and compare data from different sources.
- [New] Look up contract IDs from symbol for IB.
