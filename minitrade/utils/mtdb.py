import logging
import sqlite3
import sys
import threading
from posixpath import expanduser
from typing import Any

from attrs import asdict
from nanoid import generate
from pypika import Query, Table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                MTDB.thread_local.db = sqlite3.connect(expanduser('~/.minitrade/database/minitrade.db'))
            else:
                MTDB.thread_local.db = sqlite3.connect(expanduser('~/.minitrade/database/minitrade.pytest.db'))
            MTDB.thread_local.db.row_factory = sqlite3.Row
            return MTDB.thread_local.db

    @staticmethod
    def get_object(tablename: str, key: str, value: Any, cls=None):
        table = Table(tablename)
        stmt = Query.from_(table).select('*').where(table[key] == value)
        try:
            with MTDB.conn() as conn:
                row = conn.execute(str(stmt)).fetchone()
            return None if row is None else row if cls is None else cls.from_row(row)
        except Exception as e:
            raise RuntimeError(f'Reading object failed, cls {cls} tablename {tablename} key {key} value {value}') from e

    @staticmethod
    def get_objects(tablename: str, key: str = None, value: Any = None, cls=None):
        table = Table(tablename)
        stmt = Query.from_(table).select('*')
        if key is not None:
            stmt = stmt.where(table[key] == value)
        try:
            with MTDB.conn() as conn:
                rows = conn.execute(str(stmt)).fetchall()
            if cls is None:
                return rows
            elif cls is dict:
                return [dict(r) for r in rows]
            else:
                return [cls.from_row(r) for r in rows]
        except Exception as e:
            raise RuntimeError(
                f'Reading objects failed, cls {cls} tablename {tablename} key {key} value {value}') from e

    @staticmethod
    def save_objects(objects, tablename: str, on_conflict='error', whitelist=None):
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
    def uniqueid():
        ''' Generate a short unique id '''
        return generate('1234567890abcdef', 20)
