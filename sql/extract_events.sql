-- Product events on or before the as-of date.
SELECT
    event_id,
    customer_id,
    event_date,
    event_type
FROM events
WHERE event_date <= :as_of
ORDER BY customer_id, event_date, event_id;
