-- Subscription book used by the churn model.
-- Dates are ISO-8601 text so the same comparisons work in SQLite and in the
-- Pandas snapshot code.

PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    signup_date TEXT NOT NULL,
    segment TEXT NOT NULL,
    acquisition_channel TEXT NOT NULL,
    region TEXT NOT NULL,
    plan_tier TEXT NOT NULL
);

CREATE TABLE subscriptions (
    subscription_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT,
    monthly_fee REAL NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    txn_date TEXT NOT NULL,
    amount REAL NOT NULL,
    txn_type TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE events (
    event_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    event_date TEXT NOT NULL,
    event_type TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE INDEX idx_txn_customer_date ON transactions (customer_id, txn_date);
CREATE INDEX idx_evt_customer_date ON events (customer_id, event_date);
CREATE INDEX idx_sub_customer ON subscriptions (customer_id);
