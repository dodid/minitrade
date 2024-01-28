import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from io import StringIO
from operator import itemgetter
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from tqdm.auto import tqdm

from minitrade.datasource.base import QuoteSource
from minitrade.utils.mtdb import MTDB


class CboeFuturesQuoteSource(QuoteSource):
    '''CBOEFuturesQuoteSource retrieves historical futures data from [CBOE website](https://www.cboe.com/us/futures/market_statistics/historical_data/).

Accepted symbol format:
- CBOE futures: VX, VXF4, VXF24, VX@2024-01-17, ...

Daily bar:
- The daily OHLCV data returns historical data up to T-1.

Minute bar:
- Not available

Spot price:
- Not available

Limitations:
- Data is only available for monthly futures.
    '''

    def __init__(self, cf_method='backward', roll_day=-1):
        """
        Initialize a CboeFuturesQuoteSource object.

        Args:
            cf_method (str, optional): The method used for calculating continunous futures. Defaults to 'backward'.
            roll_day (int, optional): Day offset before expiration to roll to the next contract, should be negative. Defaults to -1.
        """
        super().__init__()
        self.cf_method = cf_method
        self.roll_day = roll_day

    def _download_futures_history(self, futures_root, num_workers=8):
        index_url = 'https://www.cboe.com/us/futures/market_statistics/historical_data/product/list/{}/'
        data_url = 'https://cdn.cboe.com/{}'
        key_fields = ['product_display', 'expire_date', 'contract_dt', 'futures_root', 'duration_type', 'hash']

        def fetch_data(file):
            now = datetime.now(ZoneInfo('US/Central'))
            try:
                cache = MTDB.get_one('CboeFuturesCache', 'hash', file['hash'])
                if cache['freeze'] or cache['update_time'] > now.replace(hour=0, minute=0, second=0, microsecond=0):
                    return itemgetter(*key_fields)(file), pd.read_csv(StringIO(cache['data']))
            except Exception:
                pass
            data = requests.get(data_url.format(file['path'])).text
            freeze = datetime.strptime(file['expire_date'], '%Y-%m-%d').date() < now.date() - timedelta(days=3)
            MTDB.save('CboeFuturesCache', file | {'data': data, 'freeze': freeze,
                      'update_time': now}, on_conflict='update')
            return itemgetter(*key_fields)(file), pd.read_csv(StringIO(data))

        def hash(file):
            return hashlib.md5(','.join(file.values()).encode()).hexdigest()

        data_files = [x for year in requests.get(index_url.format(futures_root)).json().values() for x in year]
        if not data_files:
            raise ValueError(f'No data found for {futures_root}')
        data_files = [file | {'hash': hash(file)} for file in data_files]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            data = {itemgetter(*key_fields)(file): None for file in data_files}
            futures = [executor.submit(fetch_data, file) for file in data_files]
            for future in tqdm(
                    as_completed(futures),
                    total=len(data_files),
                    desc='Downloading data', unit='file', leave=False):
                key, df = future.result()
                data[key] = df

        df = pd.concat(data).reset_index().drop(
            columns=['level_6']).rename(
            columns=dict(zip(['level_0', 'level_1', 'level_2', 'level_3', 'level_4', 'level_5'], key_fields))).sort_values(
            ['Trade Date', 'expire_date'])
        df = df[df['duration_type'] == 'M'].copy().rename(columns={'Trade Date': 'date', 'Total Volume': 'Volume'})
        return df

    def _continuous_front_month_futures(self, df, cf_method, roll_day=-1):
        """
        Calculates the continuous near-month futures prices based on the provided dataframe.

        Args:
            df (pd.DataFrame): The dataframe containing historical futures data.
            method (str | None): The method to use for adjusting the prices. Options are 'proportional' and 'backward', None for no adjustment.
            roll_day (int): Day offset before expiration to roll to the next contract, should be negative.

        Returns:
            pd.DataFrame: The dataframe with the continuous near-month futures prices.

        """
        df = self._futures_term_structure(df)
        ohlc = ['Open', 'High', 'Low', 'Close']
        out = df['M0'].copy()
        if cf_method:
            if roll_day >= 0:
                raise ValueError(f'Invalid roll day offset {roll_day}')
            expire_index = (df['M0', 'expire_date'] == df.index)
            roll_index = expire_index.shift(roll_day, fill_value=False)
            roll_fill = pd.concat([expire_index.shift(i) for i in range(roll_day+1, 1)],
                                  axis=1).any(axis=1).reindex(df.index)
            out[roll_fill] = df['M1'][roll_fill]
            if cf_method == 'proportional':
                adjust_ratio = df[roll_index]['M1', 'Close'] / df[roll_index]['M0', 'Close']
                adjust_ratio_fill = adjust_ratio.iloc[::-1].cumprod().iloc[::-1].reindex(df.index).bfill().fillna(1)
                out[ohlc] = out[ohlc].mul(adjust_ratio_fill, axis=0)
            elif cf_method == 'backward':
                adjust_delta = df[roll_index]['M1', 'Close'] - df[roll_index]['M0', 'Close']
                adjust_delta_fill = adjust_delta.iloc[::-1].cumsum().iloc[::-1].reindex(df.index).bfill().fillna(0)
                out[ohlc] = out[ohlc].add(adjust_delta_fill, axis=0)
            else:
                raise ValueError(f'Invalid adjust method {cf_method}')
        return out

    def _futures_term_structure(self, df):
        """
        Calculate the futures term structure based on the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing futures data.

        Returns:
            pd.DataFrame: The DataFrame representing the futures term structure.
        """
        df = df.copy()
        df['Month'] = df.groupby('date').cumcount()
        ohlcve = ['Open', 'High', 'Low', 'Close', 'Volume', 'expire_date']
        df = df.pivot_table(
            index='date', columns='Month', values=ohlcve, aggfunc='first')[ohlcve].swaplevel(
            axis=1).sort_index(axis=1)
        df.columns = df.columns.map(lambda x: (f'M{x[0]}', x[1]))
        df = df.reindex(zip(df.columns.get_level_values(0), ohlcve*len(df.columns.get_level_values(0))), axis=1)
        return df

    def _format_ticker(self, ticker):
        month_codes = 'FGHJKMNQUVXZ'
        try:
            root, month, year, root2, root3, expire_date = re.match(
                r'^([A-Z]+\d*)([FGHJKMNQUVXZ])(\d+)$|^([A-Z]+\d*)$|^([A-Z]+\d*)\@(\d{4}-\d{2}-\d{2})$', ticker).groups()
            root = root or root2 or root3
            month = month_codes.index(month) + 1 if month else None
            if year:
                year_now = datetime.now(ZoneInfo('US/Central')).year
                year = int(year)
                if year < 10:
                    year = year_now // 10 * 10 + year
                elif year < 100:
                    year = year_now // 100 * 100 + year
                else:
                    raise ValueError(f'Invalid year {year}')
            expire_date = expire_date or (f'{year}-{month:02d}' if year and month else None)
            return root, expire_date
        except Exception:
            raise ValueError(f'Invalid ticker {ticker}')

    def _ticker_timezone(self, ticker):
        return 'US/Central'

    def _ticker_calendar(self, ticker):
        return 'CBOE_Futures'

    def _daily_bar(self, ticker, start, end):
        root, expire_date = self._format_ticker(ticker)
        df = self._download_futures_history(root)
        if expire_date:
            a = df[df['expire_date'] == expire_date]
            df = df[df['expire_date'].str.slice(0, 7) == expire_date] if a.empty else a
            if df.empty:
                raise RuntimeError(f'No data found for {ticker}')
            df.set_index('date', inplace=True)
        else:
            df = self._continuous_front_month_futures(df, self.cf_method, self.roll_day)
        df = df.loc[start:]
        if end:
            df = df.loc[:end]
        df.index = pd.to_datetime(df.index)
        df.index.rename('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'expire_date']]
        return df

    def _minute_bar(self, ticker, start, end, interval):
        raise NotImplementedError()

    def _spot(self, tickers):
        raise NotImplementedError()

    def term_structure(self, ticker, start=None, end=None):
        """
        Retrieves the term structure data for a given futures ticker.

        Parameters:
        ticker (str): The futures ticker symbol.
        start (str, optional): The start date of the term structure data. Defaults to None.
        end (str, optional): The end date of the term structure data. Defaults to None.

        Returns:
        pandas.DataFrame: The term structure data as a DataFrame.
        """
        root, _ = self._format_ticker(ticker)
        df = self._futures_term_structure(self._download_futures_history(root))
        df = df.xs('Close', axis=1, level=1)
        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]
        df.index = pd.to_datetime(df.index)
        df.index.rename('Date', inplace=True)
        return df
