-- Customers who already existed on the as-of date.
SELECT
    customer_id,
    signup_date,
    segment,
    acquisition_channel,
    region,
    plan_tier
FROM customers
WHERE signup_date <= :as_of
ORDER BY customer_id;
