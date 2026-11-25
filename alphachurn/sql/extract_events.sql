-- Event grain for the churn pipeline.
-- Snapshot cuts and labels are applied in pandas so every as-of date
-- can reuse one extract. The join itself is plain SQL.
SELECT
    e.customer_id,
    e.event_date,
    e.event_type,
    e.amount,
    c.signup_date,
    c.plan,
    c.region,
    c.segment
FROM events AS e
INNER JOIN customers AS c
    ON c.customer_id = e.customer_id
ORDER BY e.customer_id, e.event_date;
