CREATE TABLE IF NOT EXISTS "UnionQuoteSourceConfig" (
    "name" TEXT NOT NULL,
    "config" JSON NOT NULL,
    "update_time" DATETIME NOT NULL,
    PRIMARY KEY("name")
);

ALTER TABLE
    "TradePlan" RENAME TO "TradePlan_old";

CREATE TABLE "TradePlan" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL COLLATE NOCASE,
    "strategy_file" TEXT NOT NULL,
    "ticker_css" TEXT NOT NULL,
    "market_calendar" TEXT NOT NULL,
    "market_timezone" TEXT NOT NULL,
    "data_source" TEXT,
    "backtest_start_date" TEXT NOT NULL,
    "trade_start_date" TEXT,
    "trade_time_of_day" TEXT NOT NULL,
    "entry_type" TEXT NOT NULL,
    "broker_account" TEXT,
    "commission_rate" FLOAT NOT NULL,
    "initial_cash" FLOAT NOT NULL,
    "initial_holding" JSON,
    "strict" BOOLEAN NOT NULL,
    "enabled" BOOLEAN NOT NULL,
    "create_time" DATETIME NOT NULL,
    "update_time" DATETIME,
    "broker_ticker_map" JSON,
    UNIQUE("name"),
    PRIMARY KEY("id")
);

INSERT INTO
    "TradePlan" (
        "id",
        "name",
        "strategy_file",
        "ticker_css",
        "market_calendar",
        "market_timezone",
        "data_source",
        "backtest_start_date",
        "trade_start_date",
        "trade_time_of_day",
        "entry_type",
        "broker_account",
        "commission_rate",
        "initial_cash",
        "initial_holding",
        "strict",
        "enabled",
        "create_time",
        "update_time",
        "broker_ticker_map"
    )
SELECT
    "id",
    "name",
    "strategy_file",
    "ticker_css",
    "market_calendar",
    "market_timezone",
    "data_source",
    "backtest_start_date",
    "trade_start_date",
    "trade_time_of_day",
    "entry_type",
    "broker_account",
    "commission_rate",
    "initial_cash",
    "initial_holding",
    "strict",
    "enabled",
    "create_time",
    "update_time",
    "broker_ticker_map"
FROM
    "TradePlan_old";

DROP TABLE "TradePlan_old";