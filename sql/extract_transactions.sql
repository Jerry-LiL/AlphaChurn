-- Money movement on or before the as-of date. Later rows stay out of the extract.
SELECT
    transaction_id,
    customer_id,
    txn_date,
    amount,
    txn_type
FROM transactions
WHERE txn_date <= :as_of
ORDER BY customer_id, txn_date, transaction_id;
