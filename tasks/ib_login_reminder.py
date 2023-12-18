# Default config: {"timezone": "America/New_York", "schedule": "0 9 * * Mon-Fri", "enabled": true, "notification": {"telegram": "E", "email": "E"}}

# This script is used to ensure the IB Gateway is logged in before the market opens.
# If the IB Gateway is not logged in, it sends notifications as configured.


import sys

from minitrade.broker import Broker, BrokerAccount

# Change this to exclude accounts that you don't want to check.
EXCLUSED_ACCOUNT_ALIASES = []


def check_login_status():
    ib_accounts = [account for account in BrokerAccount.list() if account.broker == 'IB']
    brokers = [Broker.get_broker(account) for account in ib_accounts if account.alias not in EXCLUSED_ACCOUNT_ALIASES]
    status = {broker.account.alias: broker.is_ready() for broker in brokers}
    not_ready = [alias for alias, ready in status.items() if not ready]
    if not_ready:
        print(f'{not_ready} is not logged in.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    check_login_status()
