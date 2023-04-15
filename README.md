## Minitrade

**[Documentation](https://dodid.github.io/minitrade/)**

- [Minitrade](#minitrade)
- [Installation](#installation)
- [Backtesting](#backtesting)
  - [Single asset strattegy](#single-asset-strattegy)
  - [Multi-asset strategy](#multi-asset-strategy)
- [Trading](#trading)
  - [Launch](#launch)
  - [Configure](#configure)
  - [IB gateway](#ib-gateway)


**Minitrade** is a personal trading system that supports both strategy backtesting and automated order execution.

It integrates with [Backtesting.py](https://github.com/kernc/backtesting.py) under the hood, and:
- Is fully compatible with Backtesting.py strategies with minor adaptions.
- Supports multi-asset portfolio and rebalancing strategies.

Or it can be used as a full trading system that:
- Automatically executes trading strategies and submits orders.
- Manages strategy execution via web console.
- Runs on very low cost machines. 

Limitations as a backtesting framework:
- Multi-asset strategy only supports long positions and market order. 

Limitations (for now) as a trading system:
- Tested only on Linux
- Support only daily bar
- Support only long positions
- Support only Interactive Brokers

On the other hand, Minitrade is intended to be easily hackable to fit individual's needs.

## Installation

Minitrade requires `python=3.10.*`


If only used as a backtesting framework:

    $ pip install minitrade

If used as a trading system, continue with the following:

    $ minitrade init

For a detailed setup guide on Ubuntu, check out [Installation](https://dodid.github.io/minitrade/install/).

## Backtesting

Minitrade uses [Backtesting.py](https://github.com/kernc/backtesting.py) as the core library for backtesting and adds the capability to implement multi-asset strategies. 

### Single asset strattegy

For single asset strategies, those written for Backtesting.py can be easily adapted to work with Minitrade. The following illustrates what changes are necessary:

```python
from minitrade.backtest import Backtest, Strategy
from minitrade.backtest.core.lib import crossover

from minitrade.backtest.core.test import SMA, GOOG


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close.df
        self.ma1 = self.I(SMA, price, 10, overlay=True)
        self.ma2 = self.I(SMA, price, 20, overlay=True)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.position().close()
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.position().close()
            self.sell()


bt = Backtest(GOOG, SmaCross, commission=.002)
stats = bt.run()
bt.plot()
```

1. Change to import from minitrade modules. Generally `backtesting` becomes `minitrade.backtest.core`.
2. Minitrade expects `Volume` data to be always avaiable. `Strategy.data` should be consisted of OHLCV.
3. Minitrade doesn't try to guess where to plot the indicators. So if you want to overlay the indicators on the main chart, set `overlay=True` explicitly.
4. `Strategy.position` is no longer a property but a function. Any occurrence of `self.position` should be changed to `self.position()`. 

That's it. Check out [compatibility](https://dodid.github.io/minitrade/compatibility/) for more details.

![plot of single-asset strategy](https://imgur.com/N3E2d6m.jpg)

Also note that some original utility functions and strategy classes only make sense for single asset strategy. Don't use those in multi-asset strategies.

### Multi-asset strategy

Minitrade extends `Backtesting.py` to support backtesting of multi-asset strategies. 

Multi-asset strategies take a 2-level column DataFrame as data input. For example, for a strategy class that intends to invest in AAPL and GOOG as a portfolio, the `self.data` should look like:

```
$ print(self.data)

                          AAPL                              GOOG 
                          Open  High  Low   Close Volume    Open  High  Low   Close Volume
Date          
2018-01-02 00:00:00-05:00 40.39 40.90 40.18 40.89 102223600 52.42 53.35 52.26 53.25 24752000
2018-01-03 00:00:00-05:00 40.95 41.43 40.82 40.88 118071600 53.22 54.31 53.16 54.12 28604000
2018-01-04 00:00:00-05:00 40.95 41.18 40.85 41.07 89738400 54.40 54.68 54.20 54.32 20092000
2018-01-05 00:00:00-05:00 41.17 41.63 41.08 41.54 94640000 54.70 55.21 54.60 55.11 25582000
2018-01-08 00:00:00-05:00 41.38 41.68 41.28 41.38 82271200 55.11 55.56 55.08 55.35 20952000
```

Like in `Backtesting.py`, `self.data` is `_Data` type that supports progressively revealing of data, and the raw DataFrame can be accessed by `self.data.df`. 

To facilitate indicator calculation, Minitrade has built-in integration with [pandas_ta](https://github.com/twopirllc/pandas-ta) as TA library. `pandas_ta` is accessible using `.ta` property of any DataFrame. Check out [here](https://github.com/twopirllc/pandas-ta#pandas-ta-dataframe-extension) for usage. `.ta` is also enhanced to support 2-level DataFrames. 

For example,

```
$ print(self.data.df.ta.sma(3))

                                 AAPL       GOOG
Date                                            
2018-01-02 00:00:00-05:00         NaN        NaN
2018-01-03 00:00:00-05:00         NaN        NaN
2018-01-04 00:00:00-05:00   40.946616  53.898000
2018-01-05 00:00:00-05:00   41.163408  54.518500
2018-01-08 00:00:00-05:00   41.331144  54.926167
```

Even simpler, `self.data.ta.sma(3)` works the same on `self.data`.


`self.I()` can take both DataFrame/Series and functions as arguments to define an indicator. If DataFrame/Series is given as input, it's expected to have exactly the same index as `self.data`. For example,

```
self.sma = self.I(self.data.df.ta.sma(3), name='SMA_3')
```

Within `Strategy.next()`, indicators are returned as type `_Array`, essentially `numpy.ndarray`, same as in `Backtesting.py`. The `.df` accessor returns either `DataFrame` or `Series` of the corresponding value. It's the caller's responsibility to know which exact type should be returned. `.s` accessor is also available but only as a syntax suger to return a `Series`. If the actual data is a DataFrame, `.s` throws a `ValueError`. 

A key addition to support multi-asset strategy is a `Strategy.alloc` attribute, which combined with `Strategy.rebalance()` API, allows to specify how cash value should be allocate among the different assets. 

Here is an example:

```python
# This strategy evenly allocates cash into the assets
# that have the top 2 highest rate-of-change every day, 
# on condition that the ROC is possitive.

class TopPositiveRoc(Strategy):
    n = 10

    def init(self):
        roc = self.data.ta.roc(self.n)
        self.roc = self.I(roc, name='ROC')

    def next(self):
        roc = self.roc.df.iloc[-1]
        self.alloc.add(
            roc.nlargest(2).index, 
            roc > 0
        ).equal_weight()
        self.rebalance()
```

`self.alloc` keeps track of what assets to be bought and how much weight is assigned to each. 

At the beginning of each `Strategy.next()` call, `self.alloc` starts empty. 

Use `alloc.add()` to add assets to a candidate pool. `alloc.add()` takes either a list-like structure or a boolean Series as input. If it's a list-like structure, all assets in the list are added to the pool. If it's a boolean Series, index items having a `True` value are added to the pool. When multiple conditions are specified in the same call, the conditions are joined by logical `AND` and the resulted assets are added the the pool. `alloc.add()` can be called multiple times which means a logical `OR` relation and add all assets involved to the pool. 

Once candidate assets are determined, Call `alloc.equal_weight()` to assign equal weight in term of value to each selected asset.

And finally, call `Strategy.rebalance()`, which will look at the current equity value, calculate the target value for each asset, calculate how many shares to buy or sell based on the current long/short positions, and generate orders that will bring the portfolio to the target allocation.

See [Writing strategy](https://dodid.github.io/minitrade/strategy/) for more details.

Run the above strategy on some DJIA components: 

![plot of multi-asset strategy](https://imgur.com/ecy6yTm.jpg)

## Trading

Trading a strategy manually is demanding. Running backtest, submitting orders, sticking to the plan despite ups and downs, and tracking performance takes not only effort, but also discipline. Minitrade makes it easy by automating the entire process.

Minitrade's trading system consists of 3 modules:
1. A scheduler, that runs strategies periodically and triggers order submission.
2. A broker gateway, that interfaces with the broker system (IB in this case) and handles the communication.
3. A web UI, that allows managing trading plans and monitoring the executions.

### Launch

To start trading, run the following:

```
# start scheduler
minitrade scheduler start

# start ibgateway
minitrade ib start 

# start web UI
minitrade web
```

See [Command line](https://dodid.github.io/minitrade/cli/) for more usages.

Use `nohup` or other tools to keep the processes running after quiting the shell.

The web UI can be accessed at: ```http://127.0.0.1:8501```

![Minitrade web UI - trading](https://imgur.com/1oOPcU7.jpg)

![Minitrade web UI - performance ](https://imgur.com/ofc9k4i.jpg)

### Configure 

Configuring the system takes a few steps:

1. Data source

    Test the data source and make sure it works. Currently only Yahoo is supported. If a proxy is needed to visit Yahoo, configure it in the UI.

2. Broker

    Put in the username and password, and give the account an alias. Note the **Account type** selected is only a hint to help remember. Whether it's paper or live depends on the account itself, rather than on what's chosen here. 

    A test connection to the broker is made before the account is saved. It verifies if the credentials are correct and a connection can be established successfully. For IB, if two-factor authentication is enabled, a notification will be sent to the mobile phone. Confirming on the mobile to finish login.

    The credentials are saved in a local database. Be sure to secure the access to the server. 

    ![Minitrade web UI - performance ](https://imgur.com/Y0lPTQx.jpg)

3. Telegram bot (required)

    Configure a Telegram bot to receive notifications and to control trader execution. Follow [the instructions](https://medium.com/geekculture/generate-telegram-token-for-bot-api-d26faf9bf064) to create a bot, and take note of the token and chat ID. Configure those in web UI. On saving the configuration, a test message will be sent. Setup is successful if the message can be received.

    After changing telegram settings, restart all minitrade processes to make the change effective.
    
4. Email provider (optional)

    Configure a **[Mailjet](https://mailjet.com)** account to receive email notifcations about backtesting and trading results. A free-tier account should be enough. Configure authorized senders in Mailjet, otherwise sending will fail. Try use different domains for senders and receivers if free email services like Hotmail, Gmail are used, otherwise, e.g., an email sending from a Hotmail address, through 3rd part servers, to the same or another Hotmail address is likely to be treated as spam and not delivered. Sending from Hotmail address to Gmail address or vice versa increases the chance of going through. On saving the configuration, a test email will be sent. Setup is successful if the email can be received.

5. Strategy

    Strategies are just Python files containing a strategy class implementation inherited from the `Strategy` class. The files can be uploaded via the UI and be made available for defining a trade plan. If a strategy class can't be found, it will show an error. If multiple strategy classes exist in a file, the one to be run should be decorated with `@entry_strategy`. To update a strategy, upload a differnt file with the same filename.

6. Trade plan

    A trade plan provides all necessary information to trade a strategy, including:
    - which strategy to run
    - the universe of assets as a list of tickers
    - which data source to get price data from
    - which timezone are the assets traded
    - which date should a backtest starts
    - which date should a trade order be generated from
    - at what time of day should a backtest run 
    - which broker account should orders be submitted through
    - how much initial cash should be invested

    The generic tickers need to be resolved to broker specific instrument IDs. Therefore a connection to broker will be made, which may trigger 2FA authentication. Pay attention to 2FA push on mobile phone if necessary.

    Once everything is defined, a test backtest dryrun will start. It should finish without error, though it will not generate any actual orders. 

    If the test run is successful, the trade plan is scheduled to run every Mon-Fri at the specified time. The time should be between market close and next market open and after when EOD market data becomes available from the selected data source. 

    Backtests can be triggered at any time without duplicate orders or any other side effects. 

    Backtesting and trading can be enabled or disabled via the UI.

    ![Minitrade web UI - trade plan ](https://imgur.com/Zhrakz5.jpg)


### IB gateway

Minitrade uses [IB's client portal API](https://www.interactivebrokers.com/en/trading/ib-api.php#client-portal-api) to submit orders. The gateway client will be downloaded and configured when `minitrade init` is run. Automated login is handled via Chrome and Selenium webdriver. 

IB disconnects a session after 24 hours or so. Minitrade checks connection status when it needs to interact with IB, i.e. when an order should be submitted or account info is retrieved via web UI. Should Minitrade initiates a connection automatically, a silent 2FA push notification would be sent to mobile phone at random times, which would be quite easy to miss and result in a login failure. After a few consecutive failed attempts, IB may lock out the account and one has to contact customer service to unlock. 

To avoid this, Minitrade only submits orders where there is already a working connection to a broker. If there is not, Minitrade sends messages via Telegram bot to notify that there are pending orders to be submitted. User should issue `/ib login` command manually to the bot to trigger a login to IB account. The 2FA push notification should be received in a few seconds and user can complete the login process on mobile phone. Once login is successful, Minitrade will be able to submit orders.

