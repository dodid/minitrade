from .fixture import *


def test_get_data_source():
    supported = QuoteSource.AVAILABLE_SOURCES
    assert 'Yahoo' in supported
    assert 'EastMoney' in supported
    for s in supported:
        source = QuoteSource.get_source(s)
        assert isinstance(source, QuoteSource)


def test_get_data_source_not_exist():
    with pytest.raises(AttributeError):
        QuoteSource.get_source('NotExist')


def test_eastmoney_get_single_ticker():
    em = QuoteSource.get_source('EastMoney')
    start = '2000-01-03'
    end, prev_close = '2022-12-10', '2022-12-09'
    df = em.daily_bar('000001', start=start, end=end)
    assert list(df.columns.levels[0]) == ['000001']
    assert list(df.columns.levels[1]) == 'Open,High,Low,Close,Volume'.split(',')
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0].strftime('%Y-%m-%d') == '2000-01-04'
    assert df.index[-1].strftime('%Y-%m-%d') == prev_close
    assert df.notna().all(axis=None) == True

    df = em.daily_bar('000001', start=start)
    assert list(df.columns.levels[0]) == ['000001']
    assert list(df.columns.levels[1]) == 'Open,High,Low,Close,Volume'.split(',')
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0].strftime('%Y-%m-%d') == '2000-01-04'
    assert df.notna().all(axis=None) == True


def test_eastmoney_get_spot():
    em = QuoteSource.get_source('EastMoney')
    s = em.spot(['000001', '000002', '000003'])
    assert type(s) == pd.Series
    assert s.index.to_list() == ['000001', '000002', '000003']
    assert s.dtype == float


def test_yahoo_get_spot():
    yahoo = QuoteSource.get_source('Yahoo')
    s = yahoo.spot(['AAPL', 'GOOG', 'META'])
    assert type(s) == pd.Series
    assert s.index.to_list() == ['AAPL', 'GOOG', 'META']
    assert s.dtype == float


def test_yahoo_get_single_ticker():
    yahoo = QuoteSource.get_source('Yahoo')
    start = '2000-01-03'
    end, prev_close = '2022-12-10', '2022-12-09'
    df = yahoo.daily_bar('AAPL', start=start, end=end)
    assert list(df.columns.levels[0]) == ['AAPL']
    assert list(df.columns.levels[1]) == 'Open,High,Low,Close,Volume'.split(',')
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0].strftime('%Y-%m-%d') == start
    assert df.index[-1].strftime('%Y-%m-%d') == prev_close
    assert df.notna().all(axis=None) == True

    # test no end date
    df = yahoo.daily_bar('AAPL', start=start)
    assert list(df.columns.levels[0]) == ['AAPL']
    assert list(df.columns.levels[1]) == 'Open,High,Low,Close,Volume'.split(',')
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0].strftime('%Y-%m-%d') == start
    assert df.notna().all(axis=None) == True


# @pytest.mark.skip(reason="take too long to test, only run manually")
def test_yahoo_get_multiple_tickers():
    yahoo = QuoteSource.get_source('Yahoo')
    tickers = ['AAPL', 'GOOG', 'META']
    start, goog_start, meta_start = '2000-01-03', '2004-08-19', '2012-05-18'
    end, prev_close = '2022-12-10', '2022-12-09'

    for tickers in [['AAPL', 'GOOG', 'META'], 'AAPL,GOOG,META']:
        df = yahoo.daily_bar(tickers, start=start, end=end)
        assert df.columns.get_level_values(0).to_list() == [*['AAPL']*5, *['GOOG']*5, *['META']*5]
        assert df.columns.get_level_values(1).to_list() == 'Open,High,Low,Close,Volume'.split(',')*3
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0].strftime('%Y-%m-%d') == meta_start
        assert df.index[-1].strftime('%Y-%m-%d') == prev_close
        assert df.notna().all(None) == True

    for tickers in [['AAPL', 'GOOG', 'META'], 'AAPL,GOOG,META']:
        df = yahoo.daily_bar(tickers, start=start, end=end, align=False)
        assert df.columns.get_level_values(0).to_list() == [*['AAPL']*5, *['GOOG']*5, *['META']*5]
        assert df.columns.get_level_values(1).to_list() == 'Open,High,Low,Close,Volume'.split(',')*3
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0].strftime('%Y-%m-%d') == start
        assert df.index[-1].strftime('%Y-%m-%d') == prev_close
        assert df['GOOG'].first_valid_index().strftime('%Y-%m-%d') == goog_start
        assert df['META'].first_valid_index().strftime('%Y-%m-%d') == meta_start
        assert df.loc[df['META'].first_valid_index():].notna().all(None) == True

    for tickers in [['AAPL', 'GOOG', 'META'], 'AAPL,GOOG,META']:
        df = yahoo.daily_bar(tickers, start=start, end=end, align=True, normalize=True)
        assert df.columns.get_level_values(0).to_list() == [*['AAPL']*5, *['GOOG']*5, *['META']*5]
        assert df.columns.get_level_values(1).to_list() == 'Open,High,Low,Close,Volume'.split(',')*3
        assert (df.xs('Close', level=1, axis=1).iloc[0] == 1).all() == True
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0].strftime('%Y-%m-%d') == meta_start
        assert df.index[-1].strftime('%Y-%m-%d') == prev_close
        assert df.notna().all(None) == True

    with pytest.raises(Exception):
        yahoo.daily_bar('AAPL , GOOG,META ')


def test_get_nasdaq_traded(clean_db):
    # @pytest.mark.skip(reason="take too long to test, only run manually")
    populate_nasdaq_traded_symbols()
    assert MTDB.get_one('NasdaqTraded', 'symbol', 'SPY') is not None
