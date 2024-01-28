BEGIN TRANSACTION;

DROP TABLE IF EXISTS "BrokerAccount";

CREATE TABLE IF NOT EXISTS "BrokerAccount" (
	"alias" TEXT NOT NULL,
	"broker" TEXT NOT NULL,
	"mode" TEXT NOT NULL,
	"username" TEXT NOT NULL,
	"password" TEXT NOT NULL,
	PRIMARY KEY("alias")
);

INSERT INTO
	"BrokerAccount" (
		"alias",
		"broker",
		"mode",
		"username",
		"password"
	)
VALUES
	('Manual', 'Manual', 'Live', '', '');

DROP TABLE IF EXISTS "IbTrade";

CREATE TABLE IF NOT EXISTS "IbTrade" (
	"execution_id" TEXT NOT NULL,
	"symbol" TEXT,
	"supports_tax_opt" TEXT,
	"side" TEXT,
	"order_description" TEXT,
	"trade_time" TEXT,
	"trade_time_r" BIGINT,
	"size" FLOAT,
	"price" TEXT,
	"submitter" TEXT,
	"exchange" TEXT,
	"commission" TEXT,
	"net_amount" FLOAT,
	"account" TEXT,
	"accountCode" TEXT,
	"company_name" TEXT,
	"contract_description_1" TEXT,
	"contract_description_2" TEXT,
	"sec_type" TEXT,
	"listing_exchange" TEXT,
	"conid" BIGINT,
	"conidEx" TEXT,
	"open_close" TEXT,
	"directed_exchange" TEXT,
	"clearing_id" TEXT,
	"clearing_name" TEXT,
	"liquidation_trade" TEXT,
	"is_event_trading" TEXT,
	"order_ref" TEXT,
	"account_allocation_name" TEXT,
	PRIMARY KEY("execution_id")
);

DROP TABLE IF EXISTS "RawOrder";

CREATE TABLE IF NOT EXISTS "RawOrder" (
	"id" TEXT NOT NULL,
	"plan_id" TEXT NOT NULL,
	"run_id" TEXT NOT NULL,
	"ticker" TEXT NOT NULL,
	"side" TEXT NOT NULL,
	"size" BIGINT NOT NULL,
	"signal_time" DATETIME NOT NULL,
	"cancelled" BOOLEAN NOT NULL,
	"broker_order_id" TEXT,
	PRIMARY KEY("id")
);

DROP TABLE IF EXISTS "TradePlan";

CREATE TABLE IF NOT EXISTS "TradePlan" (
	"id" TEXT NOT NULL,
	"name" TEXT NOT NULL COLLATE NOCASE,
	"strategy_file" TEXT NOT NULL,
	"ticker_css" TEXT NOT NULL,
	"market_calendar" TEXT NOT NULL,
	"market_timezone" TEXT NOT NULL,
	"data_source" TEXT NOT NULL,
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

DROP TABLE IF EXISTS "BacktestLog";

CREATE TABLE IF NOT EXISTS "BacktestLog" (
	"id" TEXT NOT NULL,
	"plan_id" TEXT NOT NULL,
	"plan_name" TEXT NOT NULL,
	"plan_strategy" TEXT NOT NULL,
	"plan" JSON NOT NULL,
	"data" TEXT_QUOTEDATA,
	"strategy_code" TEXT,
	"params" JSON,
	"result" TEXT_SERIES,
	"exception" TEXT,
	"stdout" TEXT,
	"stderr" TEXT,
	"log_time" DATETIME,
	PRIMARY KEY("id")
);

DROP TABLE IF EXISTS "TraderLog";

CREATE TABLE IF NOT EXISTS "TraderLog" (
	"id" TEXT NOT NULL,
	"summary" TEXT,
	"start_time" DATETIME,
	"log_time" DATETIME,
	PRIMARY KEY("id")
);

DROP TABLE IF EXISTS "OrderValidatorLog";

CREATE TABLE IF NOT EXISTS "OrderValidatorLog" (
	"id" TEXT NOT NULL,
	"order_id" TEXT NOT NULL,
	"order" JSON NOT NULL,
	"result" JSON NOT NULL,
	"exception" TEXT,
	"log_time" DATETIME,
	PRIMARY KEY("id")
);

DROP TABLE IF EXISTS "IbOrderLog";

CREATE TABLE IF NOT EXISTS "IbOrderLog" (
	"id" TEXT NOT NULL,
	"order_id" TEXT NOT NULL,
	"account_id" TEXT NOT NULL,
	"plan" JSON NOT NULL,
	"order" JSON NOT NULL,
	"iborder" JSON,
	"result" JSON,
	"exception" TEXT,
	"broker_order_id" TEXT,
	"log_time" DATETIME,
	PRIMARY KEY("id")
);

DROP TABLE IF EXISTS "IbOrder";

CREATE TABLE IF NOT EXISTS "IbOrder" (
	"acct" TEXT,
	"exchange" TEXT,
	"conidex" TEXT,
	"conid" BIGINT,
	"orderId" BIGINT NOT NULL,
	"cashCcy" TEXT,
	"sizeAndFills" TEXT,
	"orderDesc" TEXT,
	"description1" TEXT,
	"description2" TEXT,
	"ticker" TEXT,
	"secType" TEXT,
	"listingExchange" TEXT,
	"remainingQuantity" FLOAT,
	"filledQuantity" FLOAT,
	"avgPrice" FLOAT,
	"companyName" TEXT,
	"status" TEXT,
	"order_ccp_status" TEXT,
	"outsideRTH" BOOLEAN,
	"origOrderType" TEXT,
	"supportsTaxOpt" TEXT,
	"lastExecutionTime" TEXT,
	"orderType" TEXT,
	"bgColor" TEXT,
	"fgColor" TEXT,
	"price" TEXT,
	"order_ref" TEXT,
	"timeInForce" TEXT,
	"lastExecutionTime_r" BIGINT,
	"side" TEXT,
	"order_cancellation_by_system_reason" TEXT,
	"account" TEXT,
	"totalSize" FLOAT,
	PRIMARY KEY("orderId")
);

DROP TABLE IF EXISTS "ManualTrade";

CREATE TABLE IF NOT EXISTS "ManualTrade" (
	"id" TEXT NOT NULL,
	"plan_id" TEXT NOT NULL,
	"run_id" TEXT NOT NULL,
	"ticker" TEXT NOT NULL,
	"side" TEXT NOT NULL,
	"size" BIGINT NOT NULL,
	"signal_time" DATETIME NOT NULL,
	"cancelled" BOOLEAN NOT NULL,
	"broker_order_id" TEXT NOT NULL,
	"submit_time" DATETIME NOT NULL,
	"price" FLOAT,
	"commission" FLOAT,
	"trade_time" DATETIME,
	PRIMARY KEY("id")
);

CREATE TABLE IF NOT EXISTS "TaskPlan" (
	"id" TEXT NOT NULL,
	"name" TEXT NOT NULL COLLATE NOCASE,
	"task_file" TEXT NOT NULL,
	"timezone" TEXT,
	"schedule" TEXT,
	"notification" JSON,
	"enabled" BOOLEAN NOT NULL,
	"create_time" DATETIME NOT NULL,
	"update_time" DATETIME,
	unique("name"),
	PRIMARY KEY("id")
);

CREATE TABLE IF NOT EXISTS "TaskLog" (
	"id" TEXT NOT NULL,
	"plan_id" TEXT NOT NULL,
	"plan_name" TEXT NOT NULL,
	"task_code" TEXT,
	"return_value" INTEGER,
	"stdout" TEXT,
	"stderr" TEXT,
	"log_time" DATETIME,
	PRIMARY KEY("id")
);

CREATE TABLE IF NOT EXISTS "CboeFuturesCache" (
	"hash" TEXT NOT NULL,
	'product_display' TEXT,
	'expire_date' TEXT,
	'contract_dt' TEXT,
	'futures_root' TEXT,
	'duration_type' TEXT,
	"freeze" BOOLEAN NOT NULL,
	"update_time" DATETIME,
	"path" TEXT,
	"data" TEXT,
	PRIMARY KEY("hash")
);

COMMIT;