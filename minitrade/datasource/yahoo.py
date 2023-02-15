import logging

import pandas as pd
import yfinance as yf
from pypika import Query, Table

from minitrade.datasource import QuoteSource
from minitrade.utils.mtdb import MTDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuoteSourceYahoo(QuoteSource):

    @staticmethod
    def get_yahoo_proxy() -> str | None:
        ''' Get Yahoo proxy setting

        Returns
        -------
        proxy
            HTTP proxy URI or None
        '''
        from minitrade.utils.config import config
        if config.sources.yahoo.enable_proxy:
            return config.sources.yahoo.proxy
        else:
            return None

    @staticmethod
    def infer_ticker_timezone(tickers: list[str] | str) -> str | None:
        '''Return the likely timezone where the majority of tickers are traded.

        It tries to get the timezone for each ticker and returns the most frequently appeared timezone.

        Returns
        -------
        timezone : str | None
            timezone string, or None
        '''
        try:
            if isinstance(tickers, str):
                tickers = [t.strip() for t in tickers.split(',') if len(t.strip()) > 0]
            tickertimezone = Table('tickertimezone')
            stmt = Query.from_(tickertimezone).select('*').where(tickertimezone.ticker.isin(tickers))
            with MTDB.conn() as conn:
                rows = conn.execute(str(stmt)).fetchall()
            known = [row['ticker'] for row in rows]
            zones = [row['timezone'] for row in rows]
            unknowns = set(tickers) - set(known)
            for ticker in unknowns:
                try:
                    tz = yf.Ticker(ticker).get_info(proxy=QuoteSourceYahoo.get_yahoo_proxy())[
                        'exchangeTimezoneName']
                    zones.append(tz)
                    stmt = Query.into(tickertimezone).columns('ticker', 'timezone').insert(ticker, tz)
                    with MTDB.conn() as conn:
                        conn.execute(str(stmt))
                except Exception as e:
                    logger.error(e)
            if len(zones) > 0:
                # return the most frequently appeared timezone
                return max(set(zones), key=zones.count)
            else:
                raise RuntimeError('Looking up ticker timezone failed')
        except Exception as e:
            raise RuntimeError(f'Inferring ticker timezone failed for tickers {tickers}') from e

    def __init__(self, use_proxy: bool = True, proxy: str = None):
        ''' Yahoo data source constructor

        Parameters
        ----------
        use_proxy : bool
            Whether proxy setting is respected, False to ignore proxy setting and not to use proxy
        proxy: str
            Http proxy URI to override current proxy setting, None to use current setting
        '''
        if use_proxy:
            self.proxy = self.get_yahoo_proxy() if proxy is None else proxy
        else:
            self.proxy = None

    def read_daily_ohlcv(self, ticker, start='2000-01-01', end=None) -> pd.DataFrame:
        try:
            df: pd.DataFrame = yf.Ticker(ticker).history(start=start, end=end, interval='1d',
                                                         auto_adjust=True, proxy=self.proxy, timeout=10)
            df = df['Open,High,Low,Close,Volume'.split(',')]
            # df.loc[:, 'Volume'].fillna(0, inplace=True)
            # df.loc[:, 'Open,High,Low,Close'.split(',')].fillna(0, inplace=True)
            return df
        except Exception as e:
            raise RuntimeError(f'Reading OHLCV data failed for ticker={ticker} start={start} end={end}') from e
