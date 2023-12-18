# Default config: {"timezone": "America/New_York", "schedule": "0 8 * * *", "enabled": true, "notification": {"telegram": "E", "email": "N"}}

# This script is used to clean up the backtest logs in the database.

from contextlib import closing

from minitrade.utils.mtdb import MTDB

# Change this to keep more or less logs per trade plan
MAX_LOGS_PER_TRADE_PLAN = 100


def clean_backtest_logs():
    with closing(MTDB.conn()) as conn:
        rows = conn.execute('SELECT DISTINCT plan_id, plan_name FROM "BacktestLog"').fetchall()
        plans = {row['plan_id']: row['plan_name'] for row in rows}
        for plan_id, plan_name in plans.items():
            rows = conn.execute('SELECT rowid, * FROM "BacktestLog" WHERE plan_id = ? ORDER BY log_time DESC',
                                (plan_id,)).fetchall()
            if len(rows) > MAX_LOGS_PER_TRADE_PLAN:
                conn.execute('DELETE FROM "BacktestLog" WHERE plan_id = ? AND rowid < ?',
                             (plan_id, rows[MAX_LOGS_PER_TRADE_PLAN - 1]['rowid']))
                conn.commit()
                print('{:<20}: {} logs, deleted {}'.format(plan_name, len(rows), len(rows) - MAX_LOGS_PER_TRADE_PLAN))
            else:
                print('{:<20}: {} logs'.format(plan_name, len(rows)))


if __name__ == '__main__':
    clean_backtest_logs()
