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