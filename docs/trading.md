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

    Configure a Telegram bot to receive notifications and to control trader execution. It's important to do this first since setting up brokers relies on this to be functioning. Follow [the instructions](https://medium.com/geekculture/generate-telegram-token-for-bot-api-d26faf9bf064) to create a bot, and take note of the token and chat ID. Configure those in web UI. On saving the configuration, a test message will be sent. Setup is successful if the message can be received.

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

    The generic tickers need to be resolved to broker specific instrument IDs. Therefore a connection to broker will be made, which may trigger 2FA authentication. Pay attention to 2FA push on mobile phone if necessary.

    Once everything is defined, a test backtest dryrun will start. It should finish without error, though it will not generate any actual orders. 

    If the test run is successful, the trade plan is scheduled to run on every market trading day at the specified time. 

    Backtesting and trading can be enabled or disabled via the UI.

    ![Minitrade web UI - trade plan ](https://imgur.com/adrVVXq)


## IB gateway

Minitrade uses [IB's client portal API](https://www.interactivebrokers.com/en/trading/ib-api.php#client-portal-api) to submit orders. The gateway client will be downloaded and configured when `minitrade init` is run. Automated login is handled via Chrome and Selenium webdriver. 

IB disconnects a session after 24 hours or so. Minitrade checks connection status when it needs to interact with IB, i.e. when an order should be submitted or account info is retrieved via web UI. Should Minitrade initiates a connection automatically, a silent 2FA push notification would be sent to mobile phone at random times, which would be quite easy to miss and result in a login failure. After a few consecutive failed attempts, IB may lock out the account and one has to contact customer service to unlock. 

To avoid this, Minitrade only submits orders where there is already a working connection to a broker. If there is not, Minitrade sends messages via Telegram bot to notify that there are pending orders to be submitted. User should issue `/ib login` command manually to the bot to trigger a login to IB account. The 2FA push notification should be received in a few seconds and user can complete the login process on mobile phone. Once login is successful, Minitrade will be able to submit orders.

