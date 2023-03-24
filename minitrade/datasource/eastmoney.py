
import akshare as ak
import pandas as pd

from minitrade.datasource.base import QuoteSource

# Need to manually copy libmini_racer.dylib on Mac M1, see https://github.com/sqreen/PyMiniRacer/issues/143


class EastMoneyQuoteSource(QuoteSource):
    '''EastMoney data source'''

    def _daily_bar(self, ticker, start, end) -> pd.DataFrame:
        df: pd.DataFrame = ak.stock_zh_a_hist(
            symbol=ticker, period="daily",
            start_date=start.replace('-', ''),
            end_date=end.replace('-', '') if end else '20500101',   # magic number used in ak lib
            adjust='qfq')
        df = df.rename(columns={'日期': 'dt', '开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', '成交量': 'Volume'})
        df['dt'] = pd.to_datetime(df['dt'], format="%Y-%m-%d")
        df = df.set_index('dt')
        df = df.tz_localize('Asia/Shanghai')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
