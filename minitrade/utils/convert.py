
from __future__ import annotations

import json
from datetime import datetime
from io import StringIO

import attrs
from pandas import DataFrame, Series, read_csv


def attrs_to_str(attrs_obj) -> str:
    return attrs_obj if isinstance(attrs_obj, str) else json.dumps(attrs.asdict(attrs_obj), default=str)


def df_to_str(val: DataFrame | Series | str | None) -> str | None:
    return val if isinstance(val, str) else val.to_csv() if val is not None else None


def bytes_to_str(val: bytes | str | None) -> str | None:
    return val if isinstance(val, str) else val.decode("utf-8") if val is not None else None


def datetime_to_iso(dt: str | datetime | None) -> str | None:
    return dt.isoformat() if isinstance(dt, datetime) else dt


def iso_to_datetime(dt: datetime | str | None) -> datetime | None:
    return datetime.fromisoformat(dt) if isinstance(dt, str) else dt


def obj_to_str(dic) -> str | None:
    return dic if isinstance(dic, str) else json.dumps(dic, default=str)


def attrs_to_df(lst: list) -> DataFrame:
    return DataFrame([attrs.asdict(x) for x in lst])


def csv_to_df(csv: str, **kwargs) -> DataFrame | Series:
    return read_csv(StringIO(csv), **kwargs)
