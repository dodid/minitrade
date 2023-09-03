

from minitrade.broker.base import Broker, BrokerAccount
from minitrade.datasource.base import QuoteSource
from minitrade.trader.trader import TradePlan


class InteractiveBrokersQuoteSource(QuoteSource):
    '''InteractiveBrokers data source'''

    def __init__(self, plan: TradePlan | str) -> None:
        super().__init__()
        self.plan = TradePlan.get_plan(plan) if isinstance(plan, str) else plan
        self.account = BrokerAccount.get_account(plan)
        self.broker = Broker.get_broker(self.account)

    def _daily_bar(self, ticker, start, end):
        self.broker.connect()
        return self.broker.daily_bar(self.plan, ticker, start, end)

    def _spot(self, tickers):
        self.broker.connect()
        return self.broker.spot(self.plan, tickers)
