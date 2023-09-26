---
hide:
  - toc
---

# Trading

Trading a strategy manually is demanding. Running backtest, submitting orders, sticking to the plan despite ups and downs, and tracking performance takes not only effort, but also discipline. Minitrade makes it easy by automating the entire process.

Minitrade's trading system consists of 3 modules:

1. A scheduler, that runs strategies periodically and triggers order submission.
2. A broker gateway, that interfaces with the broker system (IB in this case) and handles the communication.
3. A web UI, that allows managing trading plans and monitoring the executions.

## Launch

To start trading, run the following:

```
# start scheduler
minitrade scheduler start

# start ibgateway
minitrade ib start 

# start web UI
minitrade web start
```

You can also use [the script](https://github.com/dodid/minitrade/blob/main/mtctl.sh) to simplify the management of the processes.

The web UI can be accessed at: ```http://127.0.0.1:8501```

![Minitrade web UI - trading](https://imgur.com/1oOPcU7.jpg)

![Minitrade web UI - performance ](https://imgur.com/ofc9k4i.jpg)

## Configure 

Configuring the system takes a few steps:

1. Telegram bot (required)

    Configure a Telegram bot to receive notifications and to control trader execution. **It's important to do this first since setting up brokers relies on this to be functional.** Follow [the instructions](https://medium.com/geekculture/generate-telegram-token-for-bot-api-d26faf9bf064) to create a bot, and take note of the token and chat ID. Configure those in web UI. On saving the configuration, a test message will be sent. Setup is successful if the message can be received.

    After changing telegram settings, restart all Minitrade processes to make the change effective.
    
2. Email provider (optional)

    Configure a **[Mailjet](https://mailjet.com)** account to receive email notifcations about backtesting and trading results. A free-tier account should be enough. Configure authorized senders in Mailjet, otherwise sending will fail. Try use different domains for senders and receivers if free email services like Hotmail, Gmail are used, otherwise, e.g., an email sending from a Hotmail address, through 3rd part servers, to the same or another Hotmail address is likely to be treated as spam and not delivered. Sending from Hotmail address to Gmail address or vice versa increases the chance of going through. On saving the configuration, a test email will be sent. Setup is successful if the email can be received.

    After changing email settings, restart all Minitrade processes to make the change effective.

3. Data source

    Test the data source and make sure it works. Currently Yahoo and EasyMoney are supported. If a proxy is needed to visit Yahoo, configure it in the UI. 

4. Broker

    A `Manual` broker is built in that allows to run backtest automatically and handle order placement manually.

    To set up InteractiveBrokers, put in the username and password, and give the account an alias. Note the **Account type** selected is only a hint to help remember. Whether it's paper or live depends on the account itself, rather than on what's chosen here. 

    A test connection to the broker is made before the account is saved. It verifies if the credentials are correct and a connection can be established successfully. For IB, if two-factor authentication is enabled, a notification will be sent to the mobile phone. Confirming on the mobile to finish login.

    The credentials are saved in a local database. Be sure to secure the access to the server. 

    ![Minitrade web UI - performance ](https://imgur.com/Y0lPTQx.jpg)

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

    Once everything is defined, a test backtest dryrun will start. It should finish without error, though it will not generate any actual orders. 

    If the test run is successful, the trade plan is scheduled to run on every market trading day at the specified time. 

    Backtesting and trading can be enabled or disabled via the UI.

    ![Minitrade web UI - trade plan](https://imgur.com/MofEeIT.png)

## How trading works

As depicted above, defining a trade plan takes a number of inputs. Some are self-explanatory, and some deserves some explaination.

#### Broker account

Specify which broker account you want to use to place orders. If you have configured an InteractiveBrokers account, it will show up here as default. Alternatively, you can use the built-in `Manual` account. In this case, Minitrade runs backtests automatically at scheduled moments and generates raw orders. You are supposed to place orders accordingly using broker's system that is beyond the control of Minitrade. You can backfill the actual trade price into Minitrade to track strategy performance.

#### Asset space

Specify the universe of assets for the strategy. For single asset strategy, this is just one symbol. For multi-asset strategy, this is a list of symbols separated by comma. Note the generic stock symbol, as we know it, is not usually what is traded by the broker. InteractiveBrokers uses a contract ID to uniquely identify an tradable asset. For example, "AAPL" can means any of the following contracts:

```python
APPLE INC {'conid': 265598, 'exchange': 'NASDAQ', 'isUS': True}
LS 1X AAPL {'conid': 493546048, 'exchange': 'LSEETF', 'isUS': False}
APPLE INC-CDR {'conid': 532640894, 'exchange': 'AEQLIT', 'isUS': False}
```

Minitrade let you pick which one you want to trade. The process of resolving a generic symbol to a broker specific contract ID requires an active connection to the broker. A connection will be made, which may trigger 2FA authentication, if it's not already existing. Pay attention to Telegram notification on mobile after you fill in this field.

#### Data source

Choose which data source you want to use. "Yahoo" is good for U.S. markets and "EastMoney" is for China market. If market is open, "Yahoo" returns the realtime OHLCV data (up to now) for today and historical OHLCV data for previous days.

#### Backtest start date

Specify from which date a backtest should start. This depends on how long a lead time a strategy needs to calculate indicators. If it uses a SMA of 20 days, you need to set backtest start date to a month earlier from now. Setting the date too earlier than needed makes backtesting unnecessarily slower.

#### Trade start date

Specify from which date a backtest should start generating orders. Generating orders before today is useless since you have missed the opportunities to place orders. Therefore, backtest will surpress any trade signals generated before the date.

#### Order entry type

Specify the timing or type of an order. Three order entry types are defined:

1. Trade on open (TOO)

    Use this type if you want trades to happen on market open. This is a regular market order but is submitted off regular market hours, so that it can be executed on next market open. If you use this type, make sure run backtests off regular market hours.

2. Trade on close (TOC)

    Use this type if you want trades to happen on market close. This is a [market-on-close](https://www.interactivebrokers.com/en/trading/orders/moc.php) order. It should be submitted during market hours and before market close. Usually you want to schedule backtest to run a short moment before market close to capture the latest price movement.

3. Trade regular hours (TRG)

    Use this type if you want trades to happen at earliest possible time. This is just a regular market order. If submitted during market hours, it's executed immediately. Otherwise, it's executed on next market open.

#### Backtest run time

Specify when backtest should run during a day. Minitrade makes sure that backtest runs on every day when the intended market is open. This specifies at what moment it runs. You can put in time in "HH:MM:SS" format, e.g. "10:00:00" to run at 10AM, or "9:30:01-16:00/30min" to mean backtest should run every 30 mins during market open. You can preview the exact time to confirm.

Depending on the complecity, a backtest may take seconds to minutes or longer to finish. Make sure to put enough spacing between backtests of the same strategy or different strategies. Only one backtest runs at any moment. Others ready to run will be queued. If a backtest misses its intended schedule for more than 3 minutes, it will skip and not run at all. The behavior of scheduling backtest at the exact moment of market open or market close is not well defiend. Always assume the clock can be off by a few seconds.

If you want to run backtest less frequenty than every day, e.g. rebalancing at the beginning of every month, you can handle it in the strategy. Use `self.broker.now` to find out the current date and skip processing if necessary.

#### Cash amount

Specify the initial cash you want to invest in the strategy.

#### Preexisting asset positions

Specify the initial positions that you have and want to be considered in the strategy. For example, if you already have 100 shares of SPY and 100 shares of QQQ and want to run a strategy that rebalances between them every month, you should put in "SPY:100,QQQ:100" here.

#### Backtest mode

Minitrade supports two backtest modes with very different assumptions. "Strict mode" assumes that the outcome of a backtest is repeatable, i.e., if you run a backtest for a trade plan today, the result of the backtest up to yesterday should be exactly the same as from the backtest you run yesterday. If you run the backtest tomorrow, the output up to today's date will be the same as what you get today, and so on. In this mode, backtest always runs with the parameters as given in the trade plan. If it generates an order, it assumes the order is executed successfully at the simulated price. It is unaware of what actually happends on the broker side. Since backtest runs totally independent of reality, its outcome reflecting reality only if a number of strong assumptions hold true. Notably,

1. Quotes from data source don't change. This can break if there is dividend or stock split. Minitrade aborts a backtest if it detects price changes.
2. Strategy can only issue TOC or TOO orders, since their prices are determinstic and known during backtest. 
3. Orders must be executed successfully. 

As time progressing, it's inevitable that some assumption breaks and backtest loses sync with reality. When that happens, you may want to scratch the trade plan and define a new one to bring it back in sync. This can be a lot of maintenance. The benefit is that, when it works, you can expect the actual trade performance follows closely with that of the strategy.

If some deviation from the strategy is tolerable in execution, Minitrade also support running backtest in "incremental" model. In this mode, running a backtest involves the follows:

1. Download the latest order and trade status from the broker.
2. Calculate the actual positions belonging to the strategy from initial positions and the actaul trades executed.
3. Cancel all orders, associated with the strategy, not yet submitted to the broker and all orders submitted but not yet fulfilled at the broker.
4. Run backtest, replacing the cash and initial positions as specified in the trade plan with the actual cash and positions at the moment, and setting trade_start_date to today.
5. Submit new orders if the backtest generates any.

In "incremental" mode, we asssume that the strategy is always able to make the right decisions given the latest cash and position information, and it can do it independent of previous backtest runs. This is also a strong assumption, but hopefully not too limiting if you structure the strategy properly assuming external changes can happen. The benefit is it's more robust than the "strict" mode and requires less maintenance.

You need to decide which mode is apporpriate for running your strategy.

## IB gateway

Minitrade uses [IB's client portal API](https://www.interactivebrokers.com/en/trading/ib-api.php#client-portal-api) to submit orders. The gateway client will be downloaded and configured when `minitrade init` is run. Automated login is handled via Chrome and Selenium webdriver. 

IB disconnects a session after 24 hours or so. Minitrade checks connection status when it needs to interact with IB, i.e. when an order should be submitted or account info is retrieved via web UI. Should Minitrade initiates a connection automatically, a silent 2FA push notification would be sent to mobile phone at random times, which would be quite easy to miss and result in a login failure. After a few consecutive failed attempts, IB may lock out the account and one has to contact customer service to unlock. 

To avoid this, Minitrade only submits orders where there is already a working connection to a broker. If there is not, Minitrade sends messages via Telegram bot to notify that there are pending orders to be submitted. User should issue `/ib login` command manually to the bot to trigger a login to IB account. The 2FA push notification should be received in a few seconds and user can complete the login process on mobile phone. Once login is successful, Minitrade will be able to submit orders.

