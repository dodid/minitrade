CREATE TABLE IF NOT EXISTS "UnionQuoteSourceConfig" (
    "name" TEXT NOT NULL,
    "config" JSON NOT NULL,
    "update_time" DATETIME NOT NULL,
    PRIMARY KEY("name")
);