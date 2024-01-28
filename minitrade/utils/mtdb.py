import datetime
import json
import logging
import sqlite3
import sys
import threading
from dataclasses import asdict
from io import StringIO
from posixpath import expanduser
from typing import Any

import pandas as pd
from nanoid import generate
from pypika import Order, Query, Table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def adapt_json(val):
    """Adapt object to Json string."""
    return json.dumps(val, default=str) if val is not None else None


def adapt_dataframe(val: pd.DataFrame | pd.Series | None) -> str | None:
    return val.to_csv() if val is not None else None


def adapt_bytes(val) -> str | None:
    return val.decode() if val is not None else None


def convert_date(val):
    """Convert ISO 8601 date to datetime.date object."""
    return datetime.date.fromisoformat(val.decode())


def convert_datetime(val):
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return datetime.datetime.fromisoformat(val.decode())


def convert_timestamp(val):
    """Convert Unix epoch timestamp to datetime.datetime object."""
    return datetime.datetime.fromtimestamp(int(val))


def convert_dataframe(csv: bytes) -> pd.DataFrame | pd.Series:
    return pd.read_csv(StringIO(csv.decode()), index_col=0)


def convert_dataframe2l(csv: bytes) -> pd.DataFrame:
    return pd.read_csv(StringIO(csv.decode()), index_col=0, header=[0, 1], parse_dates=True)


def convert_json(val):
    """Convert json string to object."""
    return json.loads(val.decode()) if val is not None else None


sqlite3.register_converter("date", convert_date)
sqlite3.register_converter("datetime", convert_datetime)
sqlite3.register_converter("timestamp", convert_timestamp)
sqlite3.register_converter("json", convert_json)
sqlite3.register_converter("text_series", convert_dataframe)
sqlite3.register_converter("text_quotedata", convert_dataframe2l)


def serialize_to_db(val):
    if val is None:
        return None
    elif isinstance(val, (int, float, bool, str, datetime.datetime)):
        return val
    elif isinstance(val, (pd.DataFrame, pd.Series)):
        return adapt_dataframe(val)
    elif isinstance(val, bytes):
        return adapt_bytes(val)
    else:
        return adapt_json(val)


class MTDB:

    thread_local = threading.local()

    @staticmethod
    def conn():
        ''' Return a thread local db connection. 
            Reference: https://ricardoanderegg.com/posts/python-sqlite-thread-safety/ 
        '''
        try:
            return MTDB.thread_local.db
        except Exception:
            if 'pytest' not in sys.modules:
                MTDB.thread_local.db = sqlite3.connect(
                    expanduser('~/.minitrade/database/minitrade.db'),
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            else:
                MTDB.thread_local.db = sqlite3.connect(
                    expanduser('~/.minitrade/database/minitrade.pytest.db'),
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            MTDB.thread_local.db.row_factory = sqlite3.Row
            return MTDB.thread_local.db

    @staticmethod
    def get_one(tablename: str, where_key: str, where_value: Any, *, cls=None):
        '''Return single row matching the query or None if no rows are available.'''
        table = Table(tablename)
        stmt = Query.from_(table).select('*')
        if where_key is not None:
            if where_value is None:
                stmt = stmt.where(table[where_key].isnull())
            else:
                stmt = stmt.where(table[where_key] == where_value)
        with MTDB.conn() as conn:
            row = conn.execute(str(stmt)).fetchone()
        return row if cls is None or row is None else cls(**row)

    @staticmethod
    def get_all(tablename: str, where_key: str = None, where_value: Any = None, *, where: dict = None, orderby: str |
                tuple[str, bool] = None, limit: int = None, cls=None):
        '''Return all rows matching the query as a list, or empty list if no rows are available.'''
        table = Table(tablename)
        stmt = Query.from_(table).select('*')
        if where_key is not None:
            if where_value is None:
                stmt = stmt.where(table[where_key].isnull())
            else:
                stmt = stmt.where(table[where_key] == where_value)
        if where:
            for k, v in where.items():
                if v is None:
                    stmt = stmt.where(table[k].isnull())
                else:
                    stmt = stmt.where(table[k] == v)
        if orderby is not None:
            try:
                field, ascending = orderby
                ascending = Order.asc if ascending else Order.desc
            except Exception:
                field, ascending = orderby, Order.asc
            stmt = stmt.orderby(field, order=ascending)
        if limit:
            stmt = stmt.limit(limit)
        with MTDB.conn() as conn:
            rows = conn.execute(str(stmt)).fetchall()
        return rows if cls is None else [cls(**r) for r in rows]

    @staticmethod
    def update(tablename: str, where_key: str, where_value: Any, *, values: dict):
        '''Update rows matching the query'''
        table = Table(tablename)
        stmt = Query.update(table)
        if where_key is not None:
            if where_value is None:
                stmt = stmt.where(table[where_key].isnull())
            else:
                stmt = stmt.where(table[where_key] == where_value)
        for k, v in values.items():
            stmt = stmt.set(table[k], serialize_to_db(v))
        with MTDB.conn() as conn:
            conn.execute(str(stmt))

    @staticmethod
    def save(tablename: str, objects, *, on_conflict='error', whitelist=None, blacklist=None):
        '''Save objects to ddb'''
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            data = obj if isinstance(obj, dict) else asdict(obj)
            if whitelist:
                whitelisted = {k: data.get(k, None) for k in whitelist}
                diff = set(data.keys()) - set(whitelisted.keys())
                if diff:
                    logger.warning(f'Unknonw attributes {diff} detected in {data}, not saved to {tablename}')
                data = whitelisted
            if blacklist:
                data = {k: v for k, v in data.items() if k not in blacklist}
            data = {k: serialize_to_db(v) for k, v in data.items()}
            table = Table(tablename)
            if on_conflict == 'error':
                stmt = Query.into(table).columns(*data.keys()).insert(*data.values())
            elif on_conflict == 'update':
                stmt = Query.into(table).columns(*data.keys()).replace(*data.values())
            elif on_conflict == 'ignore':
                stmt = Query.into(table).columns(*data.keys()).insert(*data.values()).ignore()
                stmt = str(stmt).replace('INSERT IGNORE', 'INSERT OR IGNORE')   # patch for sqlite syntax
            else:
                raise ValueError(f'Unknown conflict handling: {on_conflict}')
            with MTDB.conn() as conn:
                conn.execute(str(stmt))

    @staticmethod
    def delete(tablename: str, where_key: str, where_value: Any):
        '''Delete rows matching the query'''
        table = Table(tablename)
        stmt = Query.from_(table).delete()
        if where_key is not None:
            if where_value is None:
                stmt = stmt.where(table[where_key].isnull())
            else:
                stmt = stmt.where(table[where_key] == where_value)
        with MTDB.conn() as conn:
            conn.execute(str(stmt))

    @staticmethod
    def uniqueid():
        ''' Generate a short unique ID '''
        return generate('1234567890abcdef', 20)
