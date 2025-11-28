CREATE TABLE users (
    user_id UInt64,
    socdem_cluster Nullable(UInt16), -- Int16 т.к. кластеров мало
    region Nullable(UInt16)
) ENGINE = MergeTree()
ORDER BY user_id;

CREATE TABLE items (
    item_id String,
    brand_id UInt32,
    category String,
    subcategory String,
    price Float32,
    embedding Array(Float32) -- Ключевой момент
) ENGINE = MergeTree()
ORDER BY item_id;

CREATE TABLE retail_events (
    timestamp Int64, -- Если это unix epoch
    event_date Date DEFAULT toDate(timestamp / 1000000), -- Партиционирование по дням (предполагая микросекунды)
    user_id UInt64,
    item_id String,
    subdomain String,
    action_type String, -- Можно Enum, но String проще для хакатона
    os String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_date) -- Чтобы данные лежали кусками по месяцам
ORDER BY (user_id, timestamp); -- Сортировка по юзеру ускорит выборку его истории

CREATE TABLE payments (
    timestamp Int64,
    user_id UInt64,
    brand_id Nullable(UInt32),
    price Float32,
    transaction_hash String
) ENGINE = MergeTree()
ORDER BY (user_id, timestamp);

INSERT INTO retail_events  SELECT * FROM file('dataset/retail/events/0109?.pq', 'Parquet');
