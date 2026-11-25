-- Subscriptions that had started by the as-of date.
-- end_date is returned even when it falls after as_of; the label uses it and
-- the feature code does not.
SELECT
    subscription_id,
    customer_id,
    start_date,
    end_date,
    monthly_fee,
    status
FROM subscriptions
WHERE start_date <= :as_of
ORDER BY customer_id, start_date;
