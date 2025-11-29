CREATE TABLE llm_user_features_data
ENGINE = MergeTree()
ORDER BY user_id
PRIMARY KEY user_id
AS
WITH last_data_timestamp AS (
    SELECT MAX(timestamp) as ts_max from receipts r
),
recent_payments AS (
    SELECT
        p.user_id AS payment_user_id,
        sum(p.price) AS total_spent_month,
        count(p.price) AS transactions_count_month,
        groupArray((p.brand_id, p.price)) AS brand_spendings -- Для дальнейшей агрегации по брендам
    FROM payments p
    INNER JOIN llm_training_users ltu ON p.user_id = ltu.user_id
    WHERE p.timestamp >= (SELECT ts_max FROM last_data_timestamp) - 30 * 24 * 3600 * 1000000 -- Последние 30 дней
    GROUP BY p.user_id
),
recent_receipt_items AS (
    SELECT
        r.user_id AS receipt_user_id,
        groupArray(i.category) AS categories_list,
        groupArray(i.subcategory) AS subcategories_list
    FROM receipts r
    INNER JOIN items i ON r.approximate_item_id = i.item_id
    INNER JOIN llm_training_users ltu ON r.user_id = ltu.user_id
    WHERE r.timestamp >= (SELECT ts_max FROM last_data_timestamp) - 30 * 24 * 3600 * 1000000 -- Последние 30 дней
    GROUP BY r.user_id
)
SELECT
    u.user_id as user_id,
    u.socdem_cluster as socdem_cluster,
    u.region as region,
    COALESCE(rp.total_spent_month, 0) AS total_spent_month,
    COALESCE(rp.transactions_count_month, 0) AS transactions_count_month,
    rp.brand_spendings as brand_spendings, -- Array(Tuple(UInt64, Float32))
    rri.categories_list as categories_list, -- Array(String)
    rri.subcategories_list as subcategories_list -- Array(String)
FROM llm_training_users u
LEFT JOIN recent_payments rp ON u.user_id = rp.payment_user_id
LEFT JOIN recent_receipt_items rri ON u.user_id = rri.receipt_user_id;
