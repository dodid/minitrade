from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd


def calc_indicator(data_dir: pd.DataFrame,
                   func: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    out = pd.concat({ticker: func(data_dir[ticker]) for ticker in data_dir.columns.levels[0]}, axis=1)
    return out


def reserve_by_rank(s: pd.Series, rank: Tuple[int, int], candidates: pd.Series, ascending=False) -> pd.Series:
    ''' output candidates falling in the desired rank range. '''
    s_rank = s.rank(ascending=ascending)
    return (s_rank >= rank[0]+1) & (s_rank < rank[1]+1) & (candidates > 0)


def filter_by_rank(s: pd.Series, rank: Tuple[int, int], reserved: pd.Series = None, ascending=False) -> pd.Series:
    ''' keep the reserved items and make up the remaining from items falling in the desired rank range '''
    n = reserved.sum() if reserved is not None else 0
    s_rank = s.rank(ascending=ascending)
    s_candidates = (s_rank >= rank[0]+1) & (s_rank < rank[1]+1)
    if reserved is not None:
        s_candidates_exclude_reserved = s_candidates & ~reserved
        s_additional = s_rank[s_candidates_exclude_reserved].nsmallest(rank[1] - rank[0] - n)
        s_candidates = reserved | s_candidates[s_additional.index]
    return s_candidates


def filter_by_value(s: pd.Series, min_: float = None, max_: float = None) -> pd.Series:
    min_ = min_ if min_ is not None else -np.inf
    max_ = max_ if max_ is not None else np.inf
    return (s > min_) & (s < max_)


def index_to_weight(index: pd.Series):
    return index.astype(float) / index.sum() if index.sum() > 0 else pd.Series(0, index=index.index)


def index_to_list(index: pd.Series):
    return index[index].index.to_list()
