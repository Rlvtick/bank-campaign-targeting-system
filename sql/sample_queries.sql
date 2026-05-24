-- 1. Top recommended customers for campaign contact
SELECT
    customer_id,
    predicted_probability,
    score_rank,
    age,
    job,
    education,
    contact,
    month,
    poutcome,
    actual_subscription
FROM top_recommended_customers_view
ORDER BY score_rank ASC
LIMIT 25;

-- 2. Selected policy summary
SELECT *
FROM selected_policy;

-- 3. Selected versus non-selected observed performance
SELECT *
FROM selected_vs_non_selected_view
ORDER BY customer_group;

-- 4. Score distribution by probability band
SELECT *
FROM score_distribution_view
ORDER BY score_band DESC;

-- 5. Segment performance summary
SELECT *
FROM segment_performance_view
ORDER BY segment_type, average_predicted_probability DESC;

-- 6. Top-k threshold simulation results
SELECT
    split,
    contact_rate,
    targeted_customers,
    captured_subscribers,
    precision_at_k,
    capture_rate_at_k,
    lift_at_k,
    score_cutoff
FROM threshold_simulation
ORDER BY split, contact_rate;

-- 7. Selected top-policy customers by job segment
SELECT
    job,
    COUNT(*) AS selected_customers,
    SUM(actual_subscription) AS actual_subscribers,
    AVG(actual_subscription) AS observed_subscription_rate,
    AVG(predicted_probability) AS average_predicted_probability
FROM top_recommended_customers_view
GROUP BY job
ORDER BY selected_customers DESC;
