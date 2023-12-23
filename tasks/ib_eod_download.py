# Default config: {"timezone": "America/New_York", "schedule": "5 20 * * Mon-Fri", "enabled": true, "notification": {"telegram": "E", "email": "E"}}

# This script is to do a daily download of IB trades at the end of day.
# If the IB Gateway is not logged in, it sends notifications as configured.


import sys

from tabulate import tabulate

from minitrade.broker import Broker, BrokerAccount

# Change this to exclude accounts that you don't want to check.
EXCLUDED_ACCOUNT_ALIASES = []


def download_trades():
    ib_accounts = [account for account in BrokerAccount.list() if account.broker == 'IB']
    brokers = [Broker.get_broker(account) for account in ib_accounts if account.alias not in EXCLUDED_ACCOUNT_ALIASES]
    skipped = []
    for broker in brokers:
        if broker.is_ready():
            df = broker.download_trades()
            if len(df) > 0:
                df = df[['account', 'symbol', 'order_description', 'trade_time',
                         'commission', 'net_amount']].sort_values('trade_time')
                print(tabulate(df, headers='keys', showindex=False))
        else:
            skipped.append(broker.account.alias)

    if skipped:
        print(f'{skipped} skipped because they are not logged in.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    download_trades()
