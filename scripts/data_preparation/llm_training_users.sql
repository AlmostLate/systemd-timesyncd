CREATE TABLE IF NOT EXISTS llm_training_users AS
SELECT
    u.user_id as user_id,
    u.socdem_cluster as socdem_cluster,
    u.region as region
FROM users AS u
INNER JOIN (
    SELECT DISTINCT user_id FROM payments WHERE user_id IN (SELECT user_id FROM receipts)
) AS p_r ON u.user_id = p_r.user_id -- Пользователи с платежами И чеками
INNER JOIN (
    SELECT user_id, count() as cnt
    FROM payments
    GROUP BY user_id
    HAVING cnt > 15 -- Высокоактивные пользователи
) AS ha ON u.user_id = ha.user_id
-- Добавляем случайный порядок и LIMIT для выборки
ORDER BY rand()
LIMIT 15000;
