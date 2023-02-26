
from __future__ import annotations

import json
from datetime import datetime
from io import StringIO

from pandas import DataFrame, Series, read_csv


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


def csv_to_df(csv: str, **kwargs) -> DataFrame | Series:
    return read_csv(StringIO(csv), **kwargs)


def serialize_to_db(val):
    if val is None:
        return None
    elif isinstance(val, (int, float, bool, str, datetime)):
        return val
    elif isinstance(val, (DataFrame, Series)):
        return df_to_str(val)
    elif isinstance(val, bytes):
        return bytes_to_str(val)
    else:
        return obj_to_str(val)
