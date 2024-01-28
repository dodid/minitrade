
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from minitrade.datasource.base import QuoteSource
from minitrade.datasource.cboe_futures import CboeFuturesQuoteSource
from minitrade.datasource.yahoo import YahooQuoteSource
from minitrade.utils.mtdb import MTDB

from .fixture import *


class MockQuoteSource(QuoteSource):
    def __init__(self):
        pass

    def _ticker_timezone(self, ticker: str) -> str:
        return 'America/New_York'

    def _ticker_calendar(self, ticker: str) -> str:
        return 'NYSE'

    def _spot(self, tickers: list[str]) -> pd.Series:
        return pd.Series([100.0, 200.0], index=tickers)

    def _daily_bar(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.date_range(start=start, end=end)
        data = {
            'Open': [100.0] * len(dates),
            'High': [110.0] * len(dates),
            'Low': [90.0] * len(dates),
            'Close': [105.0] * len(dates),
            'Volume': [1000] * len(dates),
        }
        return pd.DataFrame(data, index=dates)

    def _minute_bar(self, ticker: str, start: str, end: str, interval: int) -> pd.DataFrame:
        dates = pd.date_range(start=start, end=end, freq=f'{interval}min')
        data = {
            'Open': [100.0] * len(dates),
            'High': [110.0] * len(dates),
            'Low': [90.0] * len(dates),
            'Close': [105.0] * len(dates),
            'Volume': [1000] * len(dates),
        }
        return pd.DataFrame(data, index=dates)


class TestQuoteSource:
    @pytest.fixture(scope='module')
    def quote_source(self):
        return MockQuoteSource()

    def test_ticker_timezone(self, quote_source):
        assert quote_source.ticker_timezone('AAPL') == 'America/New_York'

    def test_ticker_calendar(self, quote_source):
        assert quote_source.ticker_calendar('AAPL') == 'NYSE'

    def test_spot(self, quote_source):
        expected = pd.Series([100.0, 200.0], index=['AAPL', 'GOOGL'])
        assert_series_equal(quote_source.spot(['AAPL', 'GOOGL']), expected)
        assert_series_equal(quote_source.spot('AAPL,GOOGL'), expected)

    def test_daily_bar(self, quote_source):
        expected_index = pd.date_range(start='2022-01-01', end='2022-01-05')
        expected_data = {
            'Open': [100.0] * 5,
            'High': [110.0] * 5,
            'Low': [90.0] * 5,
            'Close': [105.0] * 5,
            'Volume': [1000] * 5,
        }
        single = pd.DataFrame(expected_data, index=expected_index)
        expected = pd.concat([single]*2, axis=1, keys=['AAPL', 'GOOGL'])
        assert_frame_equal(quote_source.daily_bar(['AAPL', 'GOOGL'], start='2022-01-01', end='2022-01-05'), expected)
        assert_frame_equal(quote_source.daily_bar('AAPL,GOOGL', start='2022-01-01', end='2022-01-05'), expected)

    def test_daily_bar2(self, quote_source):
        expected_index = pd.date_range(start='2022-01-01', end='2022-01-05')
        expected_data = {
            'Open': [100.0 / 105] * 5,
            'High': [110.0 / 105] * 5,
            'Low': [90.0 / 105] * 5,
            'Close': [105.0 / 105] * 5,
            'Volume': [1000] * 5,
        }
        single = pd.DataFrame(expected_data, index=expected_index)
        expected = pd.concat([single]*2, axis=1, keys=['AAPL', 'GOOGL'])
        assert_frame_equal(quote_source.daily_bar(
            ['AAPL', 'GOOGL'],
            start='2022-01-01', end='2022-01-05', normalize=True),
            expected)

    def test_minute_bar(self, quote_source):
        expected_index = pd.date_range(start='2022-01-01 09:30:00', end='2022-01-01 16:00:00', freq='1min')
        expected_data = {
            'Open': [100.0] * len(expected_index),
            'High': [110.0] * len(expected_index),
            'Low': [90.0] * len(expected_index),
            'Close': [105.0] * len(expected_index),
            'Volume': [1000] * len(expected_index),
        }
        single = pd.DataFrame(expected_data, index=expected_index)
        expected = pd.concat([single]*2, axis=1, keys=['AAPL', 'GOOGL'])
        assert_frame_equal(
            quote_source.minute_bar(
                ['AAPL', 'GOOGL'],
                start='2022-01-01 09:30:00', end='2022-01-01 16:00:00', interval=1),
            expected)
        assert_frame_equal(
            quote_source.minute_bar(
                'AAPL,GOOGL',
                start='2022-01-01 09:30:00', end='2022-01-01 16:00:00', interval=1),
            expected)


class TestYahooQuoteSource:
    @pytest.fixture(scope='module')
    def quote_source(self):
        return YahooQuoteSource()

    def test_format_ticker(self, quote_source):
        ticker = 'AAPL'
        assert quote_source._format_ticker(ticker) == ticker

    def test_ticker_timezone(self, quote_source):
        ticker = 'AAPL'
        timezone = quote_source._ticker_timezone(ticker)
        assert isinstance(timezone, str)

        ticker = 'UNKNOWN'
        with pytest.raises(KeyError):
            quote_source._ticker_timezone(ticker)

    def test_ticker_calendar(self, quote_source):
        ticker = 'AAPL'
        calendar = quote_source._ticker_calendar(ticker)
        assert isinstance(calendar, str)

    def test_daily_bar(self, quote_source):
        ticker = 'AAPL'
        start = '2022-01-01'
        end = '2022-01-10'
        for ticker in ['AAPL', 'BRK-B', '^GSPC', 'ES=F', '^TNX', 'JPY=X', '0700.HK', '000001.SS', 'BTC-USD']:
            df = quote_source._daily_bar(ticker, start, end)
            assert not df.empty
            assert df.columns.to_list() == ['Open', 'High', 'Low', 'Close', 'Volume']
            assert isinstance(df.index, pd.DatetimeIndex)
            assert df.index[0].tzinfo is None
            if ticker == 'AAPL':
                assert df.index[0].strftime('%Y-%m-%d') == '2022-01-03'
            assert df.index[-1].strftime('%Y-%m-%d') == end
            assert df.notna().all(axis=None) == True

    def test_minute_bar(self, quote_source):
        ticker = 'AAPL'
        start = (datetime.now(timezone.utc) - timedelta(days=10)).strftime('%Y-%m-%d')
        end = None
        interval = 5
        df = quote_source._minute_bar(ticker, start, end, interval)
        assert not df.empty
        assert df.index[0].tzinfo == timezone.utc

    def test_spot(self, quote_source):
        tickers = ['AAPL', 'MSFT']
        df = quote_source._spot(tickers)
        assert not df.empty
        assert isinstance(df.iloc[0], float)
        assert df.name.tzinfo == timezone.utc

    def test_invalid_interval(self, quote_source):
        ticker = 'AAPL'
        start = '2022-01-01'
        end = '2022-01-10'
        interval = 7
        with pytest.raises(ValueError):
            quote_source._minute_bar(ticker, start, end, interval)

    def test_unknown_exchange(self, quote_source):
        ticker = 'UNKNOWN'
        with patch('minitrade.datasource.yahoo.yf.Ticker') as mock_ticker:
            mock_ticker().fast_info = {'exchange': 'XXX'}
            with pytest.warns(UserWarning):
                calendar = quote_source._ticker_calendar(ticker)
            assert calendar == 'NYSE'
            mock_ticker.assert_called_with(ticker)

    def test_today(self, quote_source):
        ticker = 'AAPL'
        today = quote_source.today(ticker)
        assert isinstance(today, datetime)
        assert today.tzinfo == ZoneInfo(key='America/New_York')

    def test_is_trading_now(self, quote_source):
        ticker = 'AAPL'
        is_trading = quote_source.is_trading_now(ticker)
        assert isinstance(is_trading, bool)


class TestCboeFuturesQuoteSource:

    raw = '''
date	Open	High	Low	Close	Volume	expire_date
2023-12-12	1.00	1.00	1.00	1.00	1000	2023-12-20
2023-12-12	2.00	2.00	2.00	2.00	2000	2024-01-17
2023-12-13	1.00	1.00	1.00	1.00	1000	2023-12-20
2023-12-13	2.00	2.00	2.00	2.00	2000	2024-01-17
2023-12-14	1.00	1.00	1.00	1.00	1000	2023-12-20
2023-12-14	2.00	2.00	2.00	2.00	2000	2024-01-17
2023-12-15	1.00	1.00	1.00	1.00	1000	2023-12-20
2023-12-15	2.00	2.00	2.00	2.00	2000	2024-01-17
2023-12-18	1.00	1.00	1.00	1.00	1000	2023-12-20
2023-12-18	2.00	2.00	2.00	2.00	2000	2024-01-17
2023-12-19	1.00	1.00	1.00	1.00	1000	2023-12-20
2023-12-19	2.00	2.00	2.00	2.00	2000	2024-01-17
2023-12-20	1.00	1.00	1.00	1.00	1000	2023-12-20
2023-12-20	2.00	2.00	2.00	2.00	2000	2024-01-17
2023-12-21	3.00	3.00	3.00	3.00	3000	2024-01-17
2023-12-21	4.00	4.00	4.00	4.00	4000	2024-02-14
2023-12-22	3.00	3.00	3.00	3.00	3000	2024-01-17
2023-12-22	4.00	4.00	4.00	4.00	4000	2024-02-14
2023-12-26	3.00	3.00	3.00	3.00	3000	2024-01-17
2023-12-26	4.00	4.00	4.00	4.00	4000	2024-02-14
2023-12-27	3.00	3.00	3.00	3.00	3000	2024-01-17
2023-12-27	4.00	4.00	4.00	4.00	4000	2024-02-14
2023-12-28	3.00	3.00	3.00	3.00	3000	2024-01-17
2023-12-28	4.00	4.00	4.00	4.00	4000	2024-02-14
2023-12-29	3.00	3.00	3.00	3.00	3000	2024-01-17
2023-12-29	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-02	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-02	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-03	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-03	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-04	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-04	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-05	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-05	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-08	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-08	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-09	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-09	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-10	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-10	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-11	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-11	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-12	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-12	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-16	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-16	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-17	3.00	3.00	3.00	3.00	3000	2024-01-17
2024-01-17	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-18	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-18	6.00	6.00	6.00	6.00	6000	2024-03-20
2024-01-19	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-19	6.00	6.00	6.00	6.00	6000	2024-03-20
2024-01-22	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-22	6.00	6.00	6.00	6.00	6000	2024-03-20
2024-01-23	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-23	6.00	6.00	6.00	6.00	6000	2024-03-20
2024-01-24	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-24	6.00	6.00	6.00	6.00	6000	2024-03-20
2024-01-25	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-25	6.00	6.00	6.00	6.00	6000	2024-03-20
'''

    term_df = '''
Month,M0,M0,M0,M0,M0,M0,M1,M1,M1,M1,M1,M1
,Open,High,Low,Close,Volume,expire_date,Open,High,Low,Close,Volume,expire_date
date,,,,,,,,,,,,
2023-12-12,1.00,1.00,1.00,1.00,1000,2023-12-20,2.00,2.00,2.00,2.00,2000,2024-01-17
2023-12-13,1.00,1.00,1.00,1.00,1000,2023-12-20,2.00,2.00,2.00,2.00,2000,2024-01-17
2023-12-14,1.00,1.00,1.00,1.00,1000,2023-12-20,2.00,2.00,2.00,2.00,2000,2024-01-17
2023-12-15,1.00,1.00,1.00,1.00,1000,2023-12-20,2.00,2.00,2.00,2.00,2000,2024-01-17
2023-12-18,1.00,1.00,1.00,1.00,1000,2023-12-20,2.00,2.00,2.00,2.00,2000,2024-01-17
2023-12-19,1.00,1.00,1.00,1.00,1000,2023-12-20,2.00,2.00,2.00,2.00,2000,2024-01-17
2023-12-20,1.00,1.00,1.00,1.00,1000,2023-12-20,2.00,2.00,2.00,2.00,2000,2024-01-17
2023-12-21,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2023-12-22,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2023-12-26,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2023-12-27,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2023-12-28,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2023-12-29,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-02,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-03,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-04,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-05,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-08,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-09,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-10,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-11,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-12,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-16,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-17,3.00,3.00,3.00,3.00,3000,2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-18,5.00,5.00,5.00,5.00,5000,2024-02-14,6.00,6.00,6.00,6.00,6000,2024-03-20
2024-01-19,5.00,5.00,5.00,5.00,5000,2024-02-14,6.00,6.00,6.00,6.00,6000,2024-03-20
2024-01-22,5.00,5.00,5.00,5.00,5000,2024-02-14,6.00,6.00,6.00,6.00,6000,2024-03-20
2024-01-23,5.00,5.00,5.00,5.00,5000,2024-02-14,6.00,6.00,6.00,6.00,6000,2024-03-20
2024-01-24,5.00,5.00,5.00,5.00,5000,2024-02-14,6.00,6.00,6.00,6.00,6000,2024-03-20
2024-01-25,5.00,5.00,5.00,5.00,5000,2024-02-14,6.00,6.00,6.00,6.00,6000,2024-03-20
'''

    p1 = '''
date,Open,High,Low,Close,Volume,expire_date
2023-12-12,2.6666666667,2.6666666667,2.6666666667,2.6666666667,1000,2023-12-20
2023-12-13,2.6666666667,2.6666666667,2.6666666667,2.6666666667,1000,2023-12-20
2023-12-14,2.6666666667,2.6666666667,2.6666666667,2.6666666667,1000,2023-12-20
2023-12-15,2.6666666667,2.6666666667,2.6666666667,2.6666666667,1000,2023-12-20
2023-12-18,2.6666666667,2.6666666667,2.6666666667,2.6666666667,1000,2023-12-20
2023-12-19,2.6666666667,2.6666666667,2.6666666667,2.6666666667,1000,2023-12-20
2023-12-20,2.6666666667,2.6666666667,2.6666666667,2.6666666667,2000,2024-01-17
2023-12-21,4.00,4.00,4.00,4.00,3000,2024-01-17
2023-12-22,4.00,4.00,4.00,4.00,3000,2024-01-17
2023-12-26,4.00,4.00,4.00,4.00,3000,2024-01-17
2023-12-27,4.00,4.00,4.00,4.00,3000,2024-01-17
2023-12-28,4.00,4.00,4.00,4.00,3000,2024-01-17
2023-12-29,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-02,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-03,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-04,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-05,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-08,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-09,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-10,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-11,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-12,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-16,4.00,4.00,4.00,4.00,3000,2024-01-17
2024-01-17,4.00,4.00,4.00,4.00,4000,2024-02-14
2024-01-18,5.00,5.00,5.00,5.00,5000,2024-02-14
2024-01-19,5.00,5.00,5.00,5.00,5000,2024-02-14
2024-01-22,5.00,5.00,5.00,5.00,5000,2024-02-14
2024-01-23,5.00,5.00,5.00,5.00,5000,2024-02-14
2024-01-24,5.00,5.00,5.00,5.00,5000,2024-02-14
2024-01-25,5.00,5.00,5.00,5.00,5000,2024-02-14
'''

    p4 = '''
date	Open	High	Low	Close	Volume	expire_date
2023-12-12	2.6666666667	2.6666666667	2.6666666667	2.6666666667	1000	2023-12-20
2023-12-13	2.6666666667	2.6666666667	2.6666666667	2.6666666667	1000	2023-12-20
2023-12-14	2.6666666667	2.6666666667	2.6666666667	2.6666666667	1000	2023-12-20
2023-12-15	2.6666666667	2.6666666667	2.6666666667	2.6666666667	2000	2024-01-17
2023-12-18	2.6666666667	2.6666666667	2.6666666667	2.6666666667	2000	2024-01-17
2023-12-19	2.6666666667	2.6666666667	2.6666666667	2.6666666667	2000	2024-01-17
2023-12-20	2.6666666667	2.6666666667	2.6666666667	2.6666666667	2000	2024-01-17
2023-12-21	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-22	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-26	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-27	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-28	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-29	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-02	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-03	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-04	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-05	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-08	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-09	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-10	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-11	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-12	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-16	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-17	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-18	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-19	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-22	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-23	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-24	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-25	5.00	5.00	5.00	5.00	5000	2024-02-14
'''

    b1 = '''
date	Open	High	Low	Close	Volume	expire_date
2023-12-12	3.00	3.00	3.00	3.00	1000	2023-12-20
2023-12-13	3.00	3.00	3.00	3.00	1000	2023-12-20
2023-12-14	3.00	3.00	3.00	3.00	1000	2023-12-20
2023-12-15	3.00	3.00	3.00	3.00	1000	2023-12-20
2023-12-18	3.00	3.00	3.00	3.00	1000	2023-12-20
2023-12-19	3.00	3.00	3.00	3.00	1000	2023-12-20
2023-12-20	3.00	3.00	3.00	3.00	2000	2024-01-17
2023-12-21	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-22	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-26	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-27	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-28	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-29	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-02	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-03	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-04	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-05	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-08	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-09	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-10	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-11	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-12	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-16	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-17	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-18	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-19	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-22	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-23	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-24	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-25	5.00	5.00	5.00	5.00	5000	2024-02-14
'''

    b4 = '''
date	Open	High	Low	Close	Volume	expire_date
2023-12-12	3.00	3.00	3.00	3.00	1000	2023-12-20
2023-12-13	3.00	3.00	3.00	3.00	1000	2023-12-20
2023-12-14	3.00	3.00	3.00	3.00	1000	2023-12-20
2023-12-15	3.00	3.00	3.00	3.00	2000	2024-01-17
2023-12-18	3.00	3.00	3.00	3.00	2000	2024-01-17
2023-12-19	3.00	3.00	3.00	3.00	2000	2024-01-17
2023-12-20	3.00	3.00	3.00	3.00	2000	2024-01-17
2023-12-21	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-22	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-26	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-27	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-28	4.00	4.00	4.00	4.00	3000	2024-01-17
2023-12-29	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-02	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-03	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-04	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-05	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-08	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-09	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-10	4.00	4.00	4.00	4.00	3000	2024-01-17
2024-01-11	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-12	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-16	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-17	4.00	4.00	4.00	4.00	4000	2024-02-14
2024-01-18	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-19	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-22	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-23	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-24	5.00	5.00	5.00	5.00	5000	2024-02-14
2024-01-25	5.00	5.00	5.00	5.00	5000	2024-02-14
'''

    @pytest.fixture
    def quote_source(self):
        return CboeFuturesQuoteSource()

    def test__download_futures_history(self, quote_source):
        # Test case 1: Test downloading futures history for a specific futures root
        futures_root = "VX"
        df = quote_source._download_futures_history(futures_root)
        assert isinstance(df, pd.DataFrame)

    def test__futures_term_structure(self, quote_source):
        # Test case 1: Test futures term structure calculation
        df = pd.read_csv(StringIO(self.raw), sep='\t', index_col=None)
        expected = pd.read_csv(StringIO(self.term_df), header=[0, 1], index_col=0)
        assert_frame_equal(quote_source._futures_term_structure(df), expected)

    def test__continuous_front_month_futures(self, quote_source: CboeFuturesQuoteSource):
        # Test case 1: Test continuous near-month futures calculation with proportional adjustment
        df = pd.read_csv(StringIO(self.raw), sep='\t', index_col=None)
        expected = pd.read_csv(StringIO(self.p1), index_col=0)
        assert_frame_equal(quote_source._continuous_front_month_futures(
            df, roll_day=-1, cf_method='proportional'), expected)

        # Test case 2: Test continuous near-month futures calculation with proportional adjustment
        expected = pd.read_csv(StringIO(self.p4), sep='\t', index_col=0)
        assert_frame_equal(quote_source._continuous_front_month_futures(
            df, roll_day=-4, cf_method='proportional'), expected)

        # Test case 3: Test continuous near-month futures calculation with no adjustment
        expected = pd.read_csv(StringIO(self.term_df), header=[0, 1], index_col=0)['M0']
        assert_frame_equal(quote_source._continuous_front_month_futures(
            df, cf_method=None), expected)

        # Test case 4: Test continuous near-month futures calculation with backward adjustment
        expected = pd.read_csv(StringIO(self.b1), sep='\t', index_col=0)
        assert_frame_equal(quote_source._continuous_front_month_futures(
            df, roll_day=-1, cf_method='backward'), expected)

        # Test case 5: Test continuous near-month futures calculation with backward adjustment
        expected = pd.read_csv(StringIO(self.b4), sep='\t', index_col=0)
        assert_frame_equal(quote_source._continuous_front_month_futures(
            df, roll_day=-4, cf_method='backward'), expected)

    def test__format_ticker(self, quote_source):
        # Test case 1: Test formatting a valid ticker with year and month
        ticker = "VXZ4"
        root, expire_date = quote_source._format_ticker(ticker)
        assert root == "VX"
        assert expire_date == "2024-12"

        ticker = "VXF14"
        root, expire_date = quote_source._format_ticker(ticker)
        assert root == "VX"
        assert expire_date == "2014-01"

        # Test case 2: Test formatting a valid ticker with expire date
        ticker = "VX@2024-01-17"
        root, expire_date = quote_source._format_ticker(ticker)
        assert root == "VX"
        assert expire_date == "2024-01-17"

        # Test case 3: Test formatting a valid ticker with only root
        ticker = "VX"
        root, expire_date = quote_source._format_ticker(ticker)
        assert root == "VX"
        assert expire_date is None

        # Test case 4: Test formatting an invalid ticker
        for ticker in ["VX222", "VX@2024-01"]:
            with pytest.raises(ValueError):
                quote_source._format_ticker(ticker)

        # Test case 5: Test formatting root ticker with digit
        ticker = "AMB1"
        root, expire_date = quote_source._format_ticker(ticker)
        assert root == "AMB1"
        assert expire_date is None

        ticker = "AMB1Z3"
        root, expire_date = quote_source._format_ticker(ticker)
        assert root == "AMB1"
        assert expire_date == "2023-12"

        ticker = "AMB1@2024-01-31"
        root, expire_date = quote_source._format_ticker(ticker)
        assert root == "AMB1"
        assert expire_date == "2024-01-31"

    def test__ticker_timezone(self, quote_source):
        # Test case 1: Test getting the timezone for a ticker
        ticker = "VX"
        timezone = quote_source._ticker_timezone(ticker)
        assert timezone == "US/Central"

        # Add more test cases as needed

    def test__ticker_calendar(self, quote_source):
        # Test case 1: Test getting the calendar for a ticker
        ticker = "VX"
        calendar = quote_source._ticker_calendar(ticker)
        assert calendar == "CBOE_Futures"

    def test__daily_bar(self, quote_source):
        # Test case 1: Test getting daily bar data for a specific ticker and date range
        ticker = "VX"
        start = '2022-01-01'
        end = '2022-01-31'
        df = quote_source._daily_bar(ticker, start, end)
        assert isinstance(df, pd.DataFrame)
        assert df.columns.to_list() == ['Open', 'High', 'Low', 'Close', 'Volume', 'expire_date']
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0].strftime('%Y-%m-%d') == '2022-01-03'
        assert df.index[-1].strftime('%Y-%m-%d') == end
        assert df.index.duplicated().sum() == 0
        assert df.notna().all(axis=None) == True

        # Test case 2: Test getting daily bar data for a specific ticker and date range
        ticker = "VXF24"
        start = '2022-01-01'
        end = None
        df = quote_source._daily_bar(ticker, start, end)
        assert isinstance(df, pd.DataFrame)
        assert df.columns.to_list() == ['Open', 'High', 'Low', 'Close', 'Volume', 'expire_date']
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0].strftime('%Y-%m-%d') == '2023-04-24'
        assert df.index[-1].strftime('%Y-%m-%d') == '2024-01-17'
        assert df.index.duplicated().sum() == 0
        assert df.notna().all(axis=None) == True

    def test__minute_bar(self, quote_source):
        # Test case 1: Test getting minute bar data for a specific ticker, date range, and interval
        ticker = "VX"
        start = '2022-01-01'
        end = '2022-01-31'
        interval = 5
        with pytest.raises(NotImplementedError):
            quote_source._minute_bar(ticker, start, end, interval)

    def test__spot(self, quote_source):
        # Test case 1: Test getting spot price for a list of tickers
        tickers = ["VX", "VXF4"]
        with pytest.raises(NotImplementedError):
            quote_source._spot(tickers)

    def test_term_structure(self, quote_source: CboeFuturesQuoteSource):
        # Test case 1: Test getting futures term structure for a specific futures root
        futures_root = "VX"
        df = quote_source.term_structure(futures_root, start='2022-01-01', end='2022-01-31')
        assert isinstance(df, pd.DataFrame)
        assert df.columns.to_list() == [f'M{i}' for i in range(0, 12)]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0].strftime('%Y-%m-%d') == '2022-01-03'
        assert df.index[-1].strftime('%Y-%m-%d') == '2022-01-31'
        assert df.index.duplicated().sum() == 0
