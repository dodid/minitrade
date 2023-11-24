DROP TABLE IF EXISTS "Ticker";

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