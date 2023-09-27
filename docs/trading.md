---
hide:
  - toc
---

# Trading

Trading a strategy manually is demanding. Running backtest, submitting orders, adhering to the plan despite ups and downs in the market, and monitoring performance takes not only effort, but also discipline. Minitrade makes it easy by automating the entire process.

The trading system provided by Minitrade comprises three modules:

1. Scheduler: This module runs strategies at regular intervals and handles order submissions.
2. Broker Gateway: The broker gateway module serves as an interface between Minitrade and the broker system, specifically Interactive Brokers (IB), facilitating seamless communication.
3. Web UI: Minitrade's web user interface allows users to manage trading plans and monitor the execution of trades.

## Launch

Launch Minitrade with the following commands:

```
# start scheduler
minitrade scheduler start

# start ibgateway
minitrade ib start 

# start web UI
minitrade web start
```

You can also manage Minitrade using [the script](https://github.com/dodid/minitrade/blob/main/mtctl.sh).

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

    Test the data source and make sure it works. Currently Yahoo and EasyMoney are supported. If accessing Yahoo requires a proxy, configure it in the web UI. 

4. Broker

    The "Manual" broker allows for automated backtesting and manual order placement.

    To set up InteractiveBrokers, enter the username, password, and assign an alias to the account. The selected "Account type" is merely a reminder and does not determine whether it is a paper or live account.
    
    Before saving the account, a test connection is made to verify the correctness of the credentials and establish a successful connection. For InteractiveBrokers, if two-factor authentication is enabled, pay attention to Telegram notifications to complete the login process. 

    The credentials are stored in a local database. Ensure that server access is properly secured.

    ![Minitrade web UI - performance ](https://imgur.com/Y0lPTQx.jpg)

5. Strategy

    Strategies are Python files that contain a strategy class, which should inherit from the `Strategy` class. These files can be uploaded through the UI and be made available for defining a trade plan. If a strategy class cannot be found in the uploaded file, an error will be displayed. In case there are multiple strategy classes in a file, the one intended to be executed should be decorated with @entry_strategy. To update a strategy, upload a different file with the same filename.

6. Trade plan

    A trade plan encompasses essential information for executing a strategy, including:

    - Specifying the strategy to be executed.
    - Providing a list of tickers that make up the asset universe.
    - Selecting the data source for price data.
    - Setting the starting date for backtesting.
    - Determining the date from which trade orders should be generated.
    - Scheduling the time of day for running the backtest.
    - Assigning the broker account for order submission.
    - Defining the initial cash investment and asset holdings.

    After defining the trade plan, a test backtest dry run will be initiated. This dry run should finish error-free but will not generate any actual orders.

    Upon successful completion of the test run, the trade plan will be scheduled to execute on every market trading day at the specified time.

    Backtesting and trading functionalities can be enabled or disabled through the web UI.


## How trading works

Defining a trade plan requires various inputs as depicted in the following image.

![Minitrade web UI - trade plan](https://imgur.com/MofEeIT.png)

#### Broker account

Specify the broker account to be used for order placement. If you have configured an InteractiveBrokers account, it will be set as the default option. Alternatively, you can choose the built-in "Manual" account. With the "Manual" account, Minitrade automatically runs backtests at scheduled intervals and generates raw orders. However, it's your responsibility to place the orders through your broker's system, which is beyond the control of Minitrade. You can input the actual trade prices into Minitrade to track the performance of your strategy.

#### Asset space

Specify the universe of assets for your strategy. For a single-asset strategy, enter a single symbol. For a multi-asset strategy, provide a comma-separated list of symbols. Please note that the generic stock symbol we commonly use may not represent the actual tradable asset recognized by the broker. InteractiveBrokers uses a contract ID to uniquely identify tradable assets. For example, the symbol "AAPL" can correspond to different contracts:

```python
APPLE INC {'conid': 265598, 'exchange': 'NASDAQ', 'isUS': True}
LS 1X AAPL {'conid': 493546048, 'exchange': 'LSEETF', 'isUS': False}
APPLE INC-CDR {'conid': 532640894, 'exchange': 'AEQLIT', 'isUS': False}
```
Minitrade allows you to select the specific contract you wish to trade. The process of mapping a generic symbol to a broker-specific contract ID requires an active connection to the broker. If a connection is not already established, Minitrade will initiate one. Please pay attention to Telegram notifications on your mobile device after filling in this field.

#### Data source

Choose the desired data source. "Yahoo" is recommended for U.S. markets, while "EastMoney" is suitable for the China market. When the market is open, "Yahoo" provides real-time OHLCV data until the present moment, as well as historical data for previous days.

#### Backtest start date

Specify the starting date for the backtest. Consider the lead time required by the strategy to calculate indicators. For example, if the strategy uses a 20-day Simple Moving Average (SMA), it is advisable to set the backtest start date to a month earlier from the present date. Setting the start date too far in the past unnecessarily prolongs the backtesting process.

#### Trade start date

Specify the date from which the backtest should begin generating trade orders. Generating orders for dates prior to today is futile, as the opportunities to place orders have already passed. Therefore, the backtest will suppress any trade signals generated before the specified date.

#### Order entry type

Specify the timing or type of an order. Three order entry types are defined:

1. Trade on open (TOO)

    Use this type if you want trades to happen on market open. This is a regular market order but is submitted outside of regular market hours, so that it can be executed on next market open. Make sure you run backtests outside of regular market hours.

2. Trade on close (TOC)

    Use this type if you want trades to happen on market close. This is a [market-on-close](https://www.interactivebrokers.com/en/trading/orders/moc.php) order. It should be submitted during market hours and before market close. Usually you want to schedule backtest to run a short moment before market close to capture the latest price movement.

3. Trade regular hours (TRG)

    Use this type if you want trades to happen at earliest possible time. This is just a regular market order. If submitted during market hours, it's executed immediately. Otherwise, it's executed on next market open.

#### Backtest run time

Specify when backtest should run. Backtests can be scheduled during a day using two formats: "HH:MM:SS" and "HH:MM:SS-HH:MM:SS/Interval." For instance, "10:00:00" represents 10 AM, while "9:30:01-16:00/30min" means the backtest will run every 30 minutes during market open. You can combine both formats to define complex schedules and confirm the exact timing through a preview.

It's important to consider the complexity of the backtest, as the duration can vary from seconds to minutes or longer. To ensure smooth execution, it is recommended to leave sufficient spacing between backtests of the same strategy or different strategies. Only one backtest runs at any given time, while others are queued. If a backtest misses its intended schedule by more than 3 minutes, it will be skipped and not run at all. Additionally, keep in mind that the exact timing of backtests at market open or market close may not be well-defined, as there can be slight clock deviations.

Minitrade ensures that backtests are executed on every open trading day for the intended market. If you prefer to run backtests less frequently, such as rebalancing at the beginning of each month, you can handle this within the strategy itself. Utilize `self.broker.now` to retrieve the current date and skip processing if necessary.

#### Cash amount

Specify the amount of cash you wish to invest in the strategy as an initial investment.

#### Preexisting asset positions

Specify any existing positions that you have and want to be taken into account by the strategy. For example, if you currently hold 100 shares of SPY and 100 shares of QQQ, and you want to run a strategy that rebalances between these assets on a monthly basis, you should enter "SPY:100, QQQ:100" in this field.

#### Backtest mode

Minitrade offers two distinct backtest modes with differing assumptions. The first is "Strict mode," which assumes that a backtest's outcome is replicable. In other words, if a backtest is run for a trade plan today, the results up until yesterday will be identical to a backtest run yesterday. Similarly, running the backtest tomorrow will yield the same output up until today's date. Under this mode, the backtest always uses the parameters specified in the trade plan. It assumes successful execution of orders at simulated prices, without considering if that actually happens. The accuracy of the backtest's reflection of reality relies on a number of assumptions, notably:

1. Price quotes from the data source remain unchanged. This can break in cases of dividends or stock splits. Minitrade halts a backtest if it detects price changes.
2. The strategy can only issue "TOC" or "TOO" orders, as their prices are deterministic and known during the backtest.
3. Orders must be executed successfully.

Over time, these assumptions will break, causing the backtest to fall out of sync with reality. When this occurs, it may be necessary to remove the trade plan and create a new one to restore synchronization. Therefore this approach requires regular maintenance. But it offers the advantage of closely aligning the actual trade performance with that of the strategy.

Alternatively, if some deviation from the strategy is acceptable during execution, Minitrade also supports an "incremental" backtest mode. In this mode, the backtest follows these steps:

1. Download the latest order and trade status from the broker.
2. Calculate the actual positions associated with the strategy based on initial positions and executed trades.
3. Cancel any unsubmitted orders tied to the strategy and any submitted orders that remain unfilled at the broker.
4. Perform the backtest, replacing the cash and initial positions specified in the trade plan with the current cash and positions, and setting the trade_start_date as today.
5. Submit new orders generated by the backtest.

In the "incremental" mode, we assume that the strategy can consistently make correct decisions using the most recent cash and position information, irrespective of previous backtest runs. While this assumption is also strong, it is less restrictive if the strategy is properly designed to accommodate external changes. The benefit of this mode is increased robustness and reduced maintenance compared to the "strict" mode.

It is important to determine which mode is appropriate for running your strategy.

## IB gateway

Minitrade utilizes [IB's client portal API](https://www.interactivebrokers.com/en/trading/ib-api.php#client-portal-api) for order submission. The gateway client is downloaded and configured during the execution of `minitrade init`. Automated login is facilitated using Chrome and Selenium webdriver.

IB terminates a session after approximately 24 hours. Minitrade verifies the connection status when interacting with IB, such as when submitting an order or retrieving account information via the web UI. If Minitrade initiates a connection automatically, a silent 2FA push notification is randomly sent to the user's mobile phone. This notification may be easily missed, resulting in a login failure. Multiple consecutive failed attempts may lead to an account lockout, necessitating contact with customer service to unlock the account.

To prevent such issues, Minitrade only submits orders when a working connection to a broker is already established. If a connection does not exist, Minitrade sends notifications through a Telegram bot, informing the user of pending orders to be submitted. The user must manually issue the `/ib login` command to the bot, triggering a login to the IB account. Within a few seconds, the user should receive the 2FA push notification on their mobile phone and complete the login process. Once the login is successful, Minitrade can proceed with order submission.