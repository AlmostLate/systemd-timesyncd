CREATE TABLE IF NOT EXISTS users (
    user_id UInt64,
    socdem_cluster Nullable(Float32),
    region Nullable(Float32)
) ENGINE = MergeTree()
ORDER BY user_id;

CREATE TABLE IF NOT EXISTS brands (
    brand_id UInt64,
    embedding Array(Float32)
) ENGINE = MergeTree()
ORDER BY brand_id;

-- Товары (Retail и Marketplace)
-- поле item_source для различия
CREATE TABLE IF NOT EXISTS items (
    item_id String,
    brand_id Nullable(UInt64),
    category Nullable(String),
    subcategory Nullable(String),
    price Nullable(Float32),
    embedding Array(Float32),
    item_source LowCardinality(String) -- 'retail' или 'marketplace'
) ENGINE = MergeTree()
ORDER BY item_id;

-- События просмотров/заказов (Retail и Marketplace)
-- поле item_source для различия
CREATE TABLE IF NOT EXISTS events (
    timestamp UInt64,
    user_id UInt64,
    item_id String,
    subdomain Nullable(String),
    action_type LowCardinality(String),
    os LowCardinality(String),
    event_source LowCardinality(String) -- 'retail' или 'marketplace'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(toDateTime(timestamp / 1000000))
ORDER BY (user_id, timestamp);

CREATE TABLE IF NOT EXISTS payments (
    timestamp UInt64,
    user_id UInt64,
    brand_id Nullable(UInt64),
    price Float32,
    transaction_hash String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(toDateTime(timestamp / 1000000))
ORDER BY (user_id, timestamp);

CREATE TABLE IF NOT EXISTS receipts (
    timestamp UInt64,
    user_id UInt64,
    brand_id Nullable(UInt64),
    approximate_item_id String,
    count Float32,
    price Float32,
    transaction_hash String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(toDateTime(timestamp / 1000000))
ORDER BY (user_id, timestamp);

CREATE TABLE IF NOT EXISTS recommendations (
    user_id UInt64,
    offer_id UInt64,
    score Float32,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (user_id, created_at);
