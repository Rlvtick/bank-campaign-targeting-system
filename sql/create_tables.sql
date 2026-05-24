DROP VIEW IF EXISTS top_recommended_customers_view;
DROP VIEW IF EXISTS selected_vs_non_selected_view;
DROP VIEW IF EXISTS score_distribution_view;
DROP VIEW IF EXISTS segment_performance_view;

CREATE VIEW top_recommended_customers_view AS
SELECT
    ms.customer_id,
    ms.predicted_probability,
    ms.score_rank,
    ms.score_percentile,
    ms.selected_policy_flag,
    pc.actual_subscription,
    pc.age,
    pc.job,
    pc.marital,
    pc.education,
    pc.contact,
    pc.month,
    pc.campaign,
    pc.previous,
    pc.poutcome
FROM model_scores AS ms
LEFT JOIN processed_customers AS pc
    ON ms.customer_id = pc.customer_id
WHERE ms.selected_policy_flag = 1
ORDER BY ms.score_rank ASC;

CREATE VIEW selected_vs_non_selected_view AS
SELECT
    CASE
        WHEN ms.selected_policy_flag = 1 THEN 'Selected top policy group'
        ELSE 'Not selected'
    END AS customer_group,
    COUNT(*) AS customer_count,
    SUM(pc.actual_subscription) AS actual_subscribers,
    AVG(pc.actual_subscription) AS observed_subscription_rate,
    AVG(ms.predicted_probability) AS average_predicted_probability,
    MIN(ms.predicted_probability) AS minimum_predicted_probability,
    MAX(ms.predicted_probability) AS maximum_predicted_probability
FROM model_scores AS ms
LEFT JOIN processed_customers AS pc
    ON ms.customer_id = pc.customer_id
GROUP BY customer_group;

CREATE VIEW score_distribution_view AS
SELECT
    CASE
        WHEN predicted_probability >= 0.80 THEN '0.80 to 1.00'
        WHEN predicted_probability >= 0.60 THEN '0.60 to 0.79'
        WHEN predicted_probability >= 0.40 THEN '0.40 to 0.59'
        WHEN predicted_probability >= 0.20 THEN '0.20 to 0.39'
        ELSE '0.00 to 0.19'
    END AS score_band,
    COUNT(*) AS customer_count,
    SUM(selected_policy_flag) AS selected_customers,
    AVG(actual_subscription) AS observed_subscription_rate,
    AVG(predicted_probability) AS average_predicted_probability
FROM model_scores
GROUP BY score_band;

CREATE VIEW segment_performance_view AS
SELECT
    'job' AS segment_type,
    pc.job AS segment_value,
    COUNT(*) AS customer_count,
    SUM(ms.selected_policy_flag) AS selected_customers,
    AVG(pc.actual_subscription) AS observed_subscription_rate,
    AVG(ms.predicted_probability) AS average_predicted_probability
FROM processed_customers AS pc
LEFT JOIN model_scores AS ms
    ON pc.customer_id = ms.customer_id
GROUP BY pc.job

UNION ALL

SELECT
    'education' AS segment_type,
    pc.education AS segment_value,
    COUNT(*) AS customer_count,
    SUM(ms.selected_policy_flag) AS selected_customers,
    AVG(pc.actual_subscription) AS observed_subscription_rate,
    AVG(ms.predicted_probability) AS average_predicted_probability
FROM processed_customers AS pc
LEFT JOIN model_scores AS ms
    ON pc.customer_id = ms.customer_id
GROUP BY pc.education

UNION ALL

SELECT
    'contact' AS segment_type,
    pc.contact AS segment_value,
    COUNT(*) AS customer_count,
    SUM(ms.selected_policy_flag) AS selected_customers,
    AVG(pc.actual_subscription) AS observed_subscription_rate,
    AVG(ms.predicted_probability) AS average_predicted_probability
FROM processed_customers AS pc
LEFT JOIN model_scores AS ms
    ON pc.customer_id = ms.customer_id
GROUP BY pc.contact

UNION ALL

SELECT
    'poutcome' AS segment_type,
    pc.poutcome AS segment_value,
    COUNT(*) AS customer_count,
    SUM(ms.selected_policy_flag) AS selected_customers,
    AVG(pc.actual_subscription) AS observed_subscription_rate,
    AVG(ms.predicted_probability) AS average_predicted_probability
FROM processed_customers AS pc
LEFT JOIN model_scores AS ms
    ON pc.customer_id = ms.customer_id
GROUP BY pc.poutcome;
