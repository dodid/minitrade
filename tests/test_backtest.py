import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from minitrade.backtest.core.backtesting import Allocation

from .fixture import *


class TestAllocation:
    @pytest.fixture
    def a(self):
        return Allocation(['A', 'B', 'C', 'D'])

    def test_call_assume_first(self, a):
        assert_series_equal(a.previous_weights, pd.Series([0.0, 0.0, 0.0, 0.0], index=['A', 'B', 'C', 'D']))
        with pytest.raises(RuntimeError):
            a.weights
        with pytest.raises(RuntimeError):
            a.weights = pd.Series([0.3, 0.3, 0.4, 0.0], index=['A', 'B', 'C', 'D'])
        with pytest.raises(RuntimeError):
            a.bucket
        with pytest.raises(RuntimeError):
            a.unallocated
        with pytest.raises(RuntimeError):
            a.normalize()

    @pytest.fixture
    def b(self) -> Allocation:
        alloc = Allocation(['A', 'B', 'C', 'D'])
        alloc.assume_zero()
        return alloc

    def test_assume_zero(self, b):
        assert_series_equal(b.weights, pd.Series([0.0, 0.0, 0.0, 0.0], index=['A', 'B', 'C', 'D']))
        assert b.unallocated == 1.0
        assert len(b.bucket) == 0

    def test_individual_weight_assignment(self, b):
        b.weights['A'] = 0.5
        b.weights['B'] = 0.3
        assert_series_equal(b.weights, pd.Series([0.5, 0.3, 0.0, 0.0], index=['A', 'B', 'C', 'D']))
        assert pytest.approx(b.unallocated) == 0.2
        with pytest.raises(AssertionError):
            b.weights['C'] = 0.3
            b.weights
        with pytest.raises(AssertionError):
            b.weights['X'] = 0.1
            b.weights

    def test_entire_weight_assignment(self, b):
        b.weights = pd.Series([0.4, 0.6], index=['A', 'B'])
        assert_series_equal(b.weights, pd.Series([0.4, 0.6, 0.0, 0.0], index=['A', 'B', 'C', 'D']))
        assert b.unallocated == 0.0

    def test_bucket_creation(self, b):
        bucket = b.bucket['test']
        assert isinstance(bucket, Allocation.Bucket)

    def test_append_assets(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        assert bucket.tickers == ['A', 'B']

    def test_append_boolean_series(self, b):
        bucket = b.bucket['test']
        series = pd.Series([True, False, True, False], index=['A', 'B', 'C', 'D'])
        bucket.append(series)
        assert bucket.tickers == ['A', 'C']

    def test_append_non_boolean_series(self, b):
        bucket = b.bucket['test']
        series = pd.Series([0, 0, 0.5, 1], index=['A', 'B', 'C', 'D'])
        bucket.append(series)
        assert bucket.tickers == ['A', 'B', 'C', 'D']

    def test_append_mixed_conditions(self, b):
        bucket = b.bucket['test']
        series = pd.Series([0, 0.3, 0.5, 1], index=['A', 'B', 'C', 'D'])
        bucket.append(series, series > 0, series < 1, ['A', 'C'])
        assert bucket.tickers == ['C']

    def test_multiple_append(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.append(['C', 'D'])
        assert bucket.tickers == ['A', 'B', 'C', 'D']

    def test_multiple_append_with_duplicates_and_mixed_order(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.append(['C', 'D', 'A'])
        assert bucket.tickers == ['A', 'B', 'C', 'D']

    def test_remove_assets(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        bucket.remove(['A', 'C'])
        assert bucket.tickers == ['B']

    def test_trim_bucket(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C', 'D'])
        bucket.trim(2)
        assert bucket.tickers == ['A', 'B']

    def test_weight_empty_bucket(self, b):
        b.weights = pd.Series([0.25] * 4, index=['A', 'B', 'C', 'D'])
        bucket = b.bucket['test']
        bucket.weight_explicitly(0.5).apply('overwrite')
        assert bucket.weights.empty
        assert_series_equal(b.weights, pd.Series([0.25] * 4, index=['A', 'B', 'C', 'D']))
        bucket.weight_equally()
        assert bucket.weights.empty
        assert_series_equal(b.weights, pd.Series([0.25] * 4, index=['A', 'B', 'C', 'D']))
        bucket.weight_proportionally([])
        assert bucket.weights.empty
        assert_series_equal(b.weights, pd.Series([0.25] * 4, index=['A', 'B', 'C', 'D']))

    def test_weight_explicitly_single_value(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.weight_explicitly(0.5)
        assert_series_equal(bucket.weights, pd.Series([0.5, 0.5], index=['A', 'B']))
        for value in [-0.01, 1.01]:
            with pytest.raises(AssertionError):
                bucket.weight_explicitly(value)

    def test_weight_explicitly_list(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.weight_explicitly([0.2, 0.8])
        assert_series_equal(bucket.weights, pd.Series([0.2, 0.8], index=['A', 'B']))
        for value in [-0.01, 1.01, 0.9]:
            with pytest.raises(AssertionError):
                bucket.weight_explicitly([0.2, value])

    def test_weight_explicitly_short_list(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.weight_explicitly([0.2])
        assert_series_equal(bucket.weights, pd.Series([0.2, 0.], index=['A', 'B']))

    def test_weight_explicitly_long_list(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.weight_explicitly([0.2, 0.3, 0.4])
        assert_series_equal(bucket.weights, pd.Series([0.2, 0.3], index=['A', 'B']))

    def test_weight_explicitly_series(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        series = pd.Series([0.2, 0.8], index=['A', 'B'])
        bucket.weight_explicitly(series)
        assert_series_equal(bucket.weights, series)
        for value in [-0.01, 1.01]:
            with pytest.raises(AssertionError):
                bucket.weight_explicitly(pd.Series([0.2, value], index=['A', 'B']))

    def test_weight_explicitly_short_series(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        series = pd.Series([0.2], index=['A'])
        bucket.weight_explicitly(series)
        assert_series_equal(bucket.weights, pd.Series([0.2, 0.], index=['A', 'B']))

    def test_weight_explicitly_long_series(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        series = pd.Series([0.2, 0.3, 0.4], index=['A', 'B', 'C'])
        bucket.weight_explicitly(series)
        assert_series_equal(bucket.weights, pd.Series([0.2, 0.3], index=['A', 'B']))

    def test_weight_equally(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        bucket.weight_equally()
        assert_series_equal(bucket.weights, pd.Series([1/3] * 3, index=['A', 'B', 'C']))
        for value in [-0.01, 1.01]:
            with pytest.raises(AssertionError):
                bucket.weight_equally(value)

    def test_weight_equally_sum(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        bucket.weight_equally(sum_=0.3)
        assert_series_equal(bucket.weights, pd.Series([0.3/3] * 3, index=['A', 'B', 'C']))

    def test_weight_proportionally(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        bucket.weight_proportionally([1, 2, 3])
        assert_series_equal(bucket.weights, pd.Series([1/6, 1/3, 1/2], index=['A', 'B', 'C']))
        with pytest.raises(AssertionError):
            bucket.weight_proportionally([1, 2, -9])
        with pytest.raises(AssertionError):
            bucket.weight_proportionally([1, 2])

    def test_weight_proportionally_sum(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        bucket.weight_proportionally([1, 2, 3], sum_=0.6)
        assert_series_equal(bucket.weights, pd.Series([0.6/6, 0.6/3, 0.6/2], index=['A', 'B', 'C']))

    def test_apply_before_weight_assignment(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        with pytest.raises(RuntimeError):
            bucket.apply('accumulate')

    def test_apply_patch_method(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C', 'D'])
        bucket.weight_equally().apply()
        assert_series_equal(b.weights, pd.Series([1/4] * 4, index=['A', 'B', 'C', 'D']))
        bucket2 = b.bucket['test2']
        bucket2.append(['A', 'B'])
        bucket2.weight_explicitly(0.2).apply('update')
        assert_series_equal(b.weights, pd.Series([0.2, 0.2, 0.25, 0.25], index=['A', 'B', 'C', 'D']))

    def test_apply_replace_method(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C', 'D'])
        bucket.weight_equally().apply('overwrite')
        assert_series_equal(b.weights, pd.Series([0.25] * 4, index=['A', 'B', 'C', 'D']))
        bucket2 = b.bucket['test2']
        bucket2.append(['A', 'B'])
        bucket2.weight_explicitly(0.2).apply('overwrite')
        assert_series_equal(b.weights, pd.Series([0.2, 0.2, 0., 0.], index=['A', 'B', 'C', 'D']))

    def test_apply_sum_method(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C', 'D'])
        bucket.weight_equally(0.8).apply('accumulate')
        assert_series_equal(b.weights, pd.Series([0.2] * 4, index=['A', 'B', 'C', 'D']))
        bucket2 = b.bucket['test2']
        bucket2.append(['A', 'B'])
        bucket2.weight_equally().apply('accumulate')
        assert_series_equal(b.weights, pd.Series([0.3, 0.3, 0.2, 0.2], index=['A', 'B', 'C', 'D']))
        bucket3 = b.bucket['test3']
        bucket3.append(b.weights)
        bucket3.weight_explicitly(b.weights).apply('accumulate')
        with pytest.raises(AssertionError):
            assert_series_equal(b.weights, pd.Series([0.6, 0.6, 0.4, 0.4], index=['A', 'B', 'C', 'D']))
        b.normalize()
        assert_series_equal(b.weights, pd.Series([0.3, 0.3, 0.2, 0.2], index=['A', 'B', 'C', 'D']))

    @pytest.fixture
    def c(self) -> Allocation:
        alloc = Allocation(['A', 'B', 'C', 'D'])
        alloc.assume_zero()
        bucket = alloc.bucket['test']
        bucket.append(['A', 'B', 'C', 'D'])
        bucket.weight_equally(0.5).apply('overwrite')
        alloc._next()
        alloc.assume_previous()
        return alloc

    def test_assume_previous(self, c):
        assert_series_equal(c.previous_weights, c.weights)
        assert len(c.bucket) == 0

    def test_next(self, c):
        c._next()
        assert_series_equal(c.previous_weights, pd.Series([0.5/4] * 4, index=c.tickers))
        with pytest.raises(RuntimeError):
            c.weights

    def test_normalize(self, c):
        c.normalize()
        assert_series_equal(c.weights, pd.Series([1/4] * 4, index=c.tickers))
        assert c.unallocated == 0.0
