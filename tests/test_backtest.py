from minitrade.backtest.core.backtesting import Allocation

from .fixture import *


def test_allocation():
    tickers = list('abcde')
    alloc = Allocation(tickers)
    assert alloc.tickers == tickers
    assert alloc.selected.equals(pd.Series(False, index=tickers))
    assert alloc.current.equals(pd.Series(0, index=tickers))
    assert alloc.previous.equals(pd.Series(0, index=tickers))

    ab = pd.Series(range(5), index=tickers) < 2
    alloc.add(ab)
    assert alloc.selected.equals(pd.Series([True, True, False, False, False], index=tickers))
    assert alloc.current.equals(pd.Series(0, index=tickers))
    assert alloc.previous.equals(pd.Series(0, index=tickers))

    alloc.add(['b', 'c'])
    assert alloc.selected.equals(pd.Series([True, True, True, False, False], index=tickers))
    assert alloc.current.equals(pd.Series(0, index=tickers))
    assert alloc.previous.equals(pd.Series(0, index=tickers))

    alloc.add(pd.Series(True, index=tickers), limit=4)
    assert alloc.selected.equals(pd.Series([True, True, True, True, False], index=tickers))
    assert alloc.current.equals(pd.Series(0, index=tickers))
    assert alloc.previous.equals(pd.Series(0, index=tickers))

    alloc.drop(pd.Series(True, index=tickers), limit=3)
    assert alloc.selected.equals(pd.Series([False, True, True, True, False], index=tickers))
    assert alloc.current.equals(pd.Series(0, index=tickers))
    assert alloc.previous.equals(pd.Series(0, index=tickers))

    alloc.drop(['a', 'c'])
    assert alloc.selected.equals(pd.Series([False, True, False, True, False], index=tickers))
    assert alloc.current.equals(pd.Series(0, index=tickers))
    assert alloc.previous.equals(pd.Series(0, index=tickers))

    with pytest.raises(AttributeError):
        alloc.equal_weight(sum_=1.01)

    with pytest.raises(AttributeError):
        alloc.equal_weight(sum_=-0.01)

    alloc.equal_weight()
    assert alloc.selected.equals(pd.Series([False, True, False, True, False], index=tickers))
    assert alloc.current.equals(pd.Series([0, 0.5, 0, 0.5, 0], index=tickers))
    assert alloc.previous.equals(pd.Series(0, index=tickers))

    alloc.equal_weight(sum_=0.8)
    assert alloc.selected.equals(pd.Series([False, True, False, True, False], index=tickers))
    assert alloc.current.equals(pd.Series([0, 0.4, 0, 0.4, 0], index=tickers))
    assert alloc.previous.equals(pd.Series(0, index=tickers))

    alloc._next()
    assert alloc.selected.equals(pd.Series(False, index=tickers))
    assert alloc.current.equals(pd.Series(0, index=tickers))
    assert alloc.previous.equals(pd.Series([0, 0.4, 0, 0.4, 0], index=tickers))
