-- =============================================================================
-- FinTech Fraud Detection & Risk Analytics SQL Suite
-- Engine Compatibility: Spark SQL, Trino, Databricks Delta Lake
-- Author: Prabhat Sharma (https://github.com/prabhat84sharma-ui)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Query 1: Top 10 Highest-Risk Transactions per Merchant Category (Dense Window Ranking)
-- Identifies peak severity transactions within each merchant domain over the past 24 hours.
-- -----------------------------------------------------------------------------
WITH ranked_transactions AS (
    SELECT
        transaction_id,
        customer_id,
        merchant_name,
        merchant_category,
        channel,
        amount,
        currency,
        risk_score,
        flagged_rule_codes,
        event_time,
        DENSE_RANK() OVER (
            PARTITION BY merchant_category 
            ORDER BY risk_score DESC, amount DESC
        ) AS category_risk_rank
    FROM delta.`data/delta/silver/transactions`
    WHERE event_time >= CURRENT_TIMESTAMP() - INTERVAL 1 DAY
)
SELECT
    merchant_category,
    category_risk_rank,
    transaction_id,
    customer_id,
    merchant_name,
    channel,
    amount,
    currency,
    risk_score,
    flagged_rule_codes,
    event_time
FROM ranked_transactions
WHERE category_risk_rank <= 10
ORDER BY merchant_category ASC, risk_score DESC;


-- -----------------------------------------------------------------------------
-- Query 2: Merchant Category Fraud Exposure & Chargeback Risk Matrix
-- Summarizes total volume, flagged volume, and fraud rate across categories.
-- -----------------------------------------------------------------------------
SELECT
    merchant_category,
    COUNT(transaction_id) AS total_transactions,
    ROUND(SUM(amount), 2) AS total_authorized_volume_usd,
    COUNT(CASE WHEN risk_score >= 80 THEN 1 END) AS quarantined_fraud_count,
    ROUND(SUM(CASE WHEN risk_score >= 80 THEN amount ELSE 0.0 END), 2) AS quarantined_fraud_volume_usd,
    ROUND(
        (COUNT(CASE WHEN risk_score >= 80 THEN 1 END) * 100.0) / COUNT(transaction_id), 
        2
    ) AS fraud_rate_pct,
    ROUND(AVG(risk_score), 2) AS avg_merchant_risk_score
FROM delta.`data/delta/silver/transactions`
GROUP BY merchant_category
ORDER BY quarantined_fraud_volume_usd DESC, fraud_rate_pct DESC;


-- -----------------------------------------------------------------------------
-- Query 3: Impossible Travel Corridors & Cross-Border Velocity Hotspots
-- Pinpoints origin-destination country pairs with supersonic or impossible travel speeds.
-- -----------------------------------------------------------------------------
SELECT
    COALESCE(prev_ip_country, 'UNKNOWN') AS origin_country,
    ip_country AS destination_country,
    COUNT(transaction_id) AS impossible_travel_incidents,
    ROUND(AVG(haversine_distance_km), 1) AS avg_distance_km,
    ROUND(MAX(travel_speed_kmh), 1) AS peak_recorded_speed_kmh,
    ROUND(SUM(amount), 2) AS intercepted_at_risk_amount
FROM delta.`data/delta/silver/transactions`
WHERE is_impossible_travel = 1
GROUP BY prev_ip_country, ip_country
ORDER BY impossible_travel_incidents DESC, intercepted_at_risk_amount DESC;


-- -----------------------------------------------------------------------------
-- Query 4: Rolling 7-Day Cumulative Fraud Loss and Mitigation Rate
-- Evaluates daily financial exposure and automated defense interception rates.
-- -----------------------------------------------------------------------------
WITH daily_metrics AS (
    SELECT
        CAST(event_time AS DATE) AS transaction_date,
        COUNT(transaction_id) AS total_txns,
        SUM(amount) AS gross_volume,
        SUM(CASE WHEN risk_score >= 80 THEN amount ELSE 0.0 END) AS prevented_fraud_loss,
        SUM(CASE WHEN is_fraud_label = 1 AND risk_score < 60 THEN amount ELSE 0.0 END) AS unmitigated_leakage_loss
    FROM delta.`data/delta/silver/transactions`
    GROUP BY CAST(event_time AS DATE)
)
SELECT
    transaction_date,
    total_txns,
    ROUND(gross_volume, 2) AS daily_gross_volume,
    ROUND(prevented_fraud_loss, 2) AS daily_prevented_loss,
    ROUND(unmitigated_leakage_loss, 2) AS daily_leakage_loss,
    ROUND(
        SUM(prevented_fraud_loss) OVER (
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2
    ) AS rolling_7d_prevented_fraud_loss,
    ROUND(
        (SUM(prevented_fraud_loss) OVER (ORDER BY transaction_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) * 100.0) /
        NULLIF(SUM(gross_volume) OVER (ORDER BY transaction_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 0),
        2
    ) AS rolling_7d_mitigation_rate_pct
FROM daily_metrics
ORDER BY transaction_date DESC;


-- -----------------------------------------------------------------------------
-- Query 5: Customer Behavioral Risk Trajectory & Rapid Burst Frequency
-- Detects customers experiencing rapid account takeover bursts within short spans.
-- -----------------------------------------------------------------------------
SELECT
    customer_id,
    COUNT(transaction_id) AS total_txns,
    ROUND(SUM(amount), 2) AS total_amount,
    MAX(txn_count_5m) AS peak_5m_burst_count,
    ROUND(MAX(amount_to_avg_ratio), 1) AS max_amount_spike_multiplier,
    MAX(risk_score) AS max_risk_score,
    ARRAY_DISTINCT(FLATTEN(COLLECT_LIST(flagged_rule_codes))) AS triggered_rules_summary,
    MIN(event_time) AS first_observed_txn,
    MAX(event_time) AS latest_observed_txn
FROM delta.`data/delta/silver/transactions`
GROUP BY customer_id
HAVING MAX(risk_score) >= 60 OR MAX(txn_count_5m) >= 3
ORDER BY max_risk_score DESC, peak_5m_burst_count DESC;


-- -----------------------------------------------------------------------------
-- Query 6: Diurnal vs. Nocturnal Fraud Vulnerability Breakdown (Hourly Distribution)
-- Highlights hours of the day with disproportionate fraudulent activity.
-- -----------------------------------------------------------------------------
SELECT
    HOUR(event_time) AS hour_of_day,
    CASE 
        WHEN HOUR(event_time) BETWEEN 1 AND 4 THEN 'Nocturnal (High Alert)'
        WHEN HOUR(event_time) BETWEEN 5 AND 11 THEN 'Morning'
        WHEN HOUR(event_time) BETWEEN 12 AND 17 THEN 'Afternoon'
        ELSE 'Evening'
    END AS day_part,
    COUNT(transaction_id) AS total_volume,
    COUNT(CASE WHEN risk_score >= 80 THEN 1 END) AS high_risk_fraud_txns,
    ROUND(
        (COUNT(CASE WHEN risk_score >= 80 THEN 1 END) * 100.0) / COUNT(transaction_id), 
        2
    ) AS fraud_share_pct,
    ROUND(AVG(amount), 2) AS avg_ticket_size
FROM delta.`data/delta/silver/transactions`
GROUP BY HOUR(event_time)
ORDER BY hour_of_day ASC;


-- -----------------------------------------------------------------------------
-- Query 7: Channel-Specific Fraud Dispersion Matrix
-- Contrasts channel vulnerability (Mobile App, Web, POS Terminal, ATM).
-- -----------------------------------------------------------------------------
SELECT
    channel,
    device_type,
    COUNT(transaction_id) AS txn_count,
    ROUND(SUM(amount), 2) AS channel_volume,
    COUNT(CASE WHEN risk_score >= 80 THEN 1 END) AS quarantined_count,
    ROUND(AVG(risk_score), 2) AS mean_risk_score,
    ROUND(
        (COUNT(CASE WHEN risk_score >= 80 THEN 1 END) * 100.0) / COUNT(transaction_id), 
        2
    ) AS channel_quarantine_rate_pct
FROM delta.`data/delta/silver/transactions`
GROUP BY channel, device_type
ORDER BY channel_quarantine_rate_pct DESC, channel_volume DESC;


-- -----------------------------------------------------------------------------
-- Query 8: Rapid-Fire Card Testing Cluster Identification (Micro-Charges)
-- Identifies card brute-forcing clusters (< $5 within <= 60s).
-- -----------------------------------------------------------------------------
SELECT
    customer_id,
    card_id,
    COUNT(transaction_id) AS consecutive_micro_txns,
    ROUND(SUM(amount), 2) AS total_tested_amount,
    ROUND(AVG(time_since_prev_txn_sec), 1) AS avg_inter_transaction_sec,
    COLLECT_SET(merchant_name) AS target_merchants,
    MIN(event_time) AS test_cluster_started,
    MAX(event_time) AS test_cluster_ended
FROM delta.`data/delta/silver/transactions`
WHERE amount <= 5.0 AND time_since_prev_txn_sec <= 60
GROUP BY customer_id, card_id
HAVING COUNT(transaction_id) >= 2
ORDER BY consecutive_micro_txns DESC;


-- -----------------------------------------------------------------------------
-- Query 9: Deterministic Rule Efficacy, Hit Rate, and Trigger Overlap
-- Analyzes which deterministic rules fire most frequently and their correlation with fraud labels.
-- -----------------------------------------------------------------------------
SELECT
    'GEO_IMPOSSIBLE_TRAVEL' AS rule_name,
    COUNT(CASE WHEN rule_impossible_travel = 1 THEN 1 END) AS trigger_count,
    ROUND((COUNT(CASE WHEN rule_impossible_travel = 1 THEN 1 END) * 100.0) / COUNT(*), 2) AS rule_coverage_pct,
    COUNT(CASE WHEN rule_impossible_travel = 1 AND is_fraud_label = 1 THEN 1 END) AS true_positives,
    ROUND(
        (COUNT(CASE WHEN rule_impossible_travel = 1 AND is_fraud_label = 1 THEN 1 END) * 100.0) /
        NULLIF(COUNT(CASE WHEN rule_impossible_travel = 1 THEN 1 END), 0), 2
    ) AS precision_pct
FROM delta.`data/delta/silver/transactions`
UNION ALL
SELECT
    'AMOUNT_SPIKE_5X',
    COUNT(CASE WHEN rule_amount_spike = 1 THEN 1 END),
    ROUND((COUNT(CASE WHEN rule_amount_spike = 1 THEN 1 END) * 100.0) / COUNT(*), 2),
    COUNT(CASE WHEN rule_amount_spike = 1 AND is_fraud_label = 1 THEN 1 END),
    ROUND(
        (COUNT(CASE WHEN rule_amount_spike = 1 AND is_fraud_label = 1 THEN 1 END) * 100.0) /
        NULLIF(COUNT(CASE WHEN rule_amount_spike = 1 THEN 1 END), 0), 2
    )
FROM delta.`data/delta/silver/transactions`
UNION ALL
SELECT
    'VELOCITY_BURST_5M',
    COUNT(CASE WHEN rule_velocity_burst = 1 THEN 1 END),
    ROUND((COUNT(CASE WHEN rule_velocity_burst = 1 THEN 1 END) * 100.0) / COUNT(*), 2),
    COUNT(CASE WHEN rule_velocity_burst = 1 AND is_fraud_label = 1 THEN 1 END),
    ROUND(
        (COUNT(CASE WHEN rule_velocity_burst = 1 AND is_fraud_label = 1 THEN 1 END) * 100.0) /
        NULLIF(COUNT(CASE WHEN rule_velocity_burst = 1 THEN 1 END), 0), 2
    )
FROM delta.`data/delta/silver/transactions`
UNION ALL
SELECT
    'HIGH_RISK_JURISDICTION',
    COUNT(CASE WHEN rule_high_risk_geo = 1 THEN 1 END),
    ROUND((COUNT(CASE WHEN rule_high_risk_geo = 1 THEN 1 END) * 100.0) / COUNT(*), 2),
    COUNT(CASE WHEN rule_high_risk_geo = 1 AND is_fraud_label = 1 THEN 1 END),
    ROUND(
        (COUNT(CASE WHEN rule_high_risk_geo = 1 AND is_fraud_label = 1 THEN 1 END) * 100.0) /
        NULLIF(COUNT(CASE WHEN rule_high_risk_geo = 1 THEN 1 END), 0), 2
    )
FROM delta.`data/delta/silver/transactions`;


-- -----------------------------------------------------------------------------
-- Query 10: High-Risk Jurisdiction Inflow Matrix & Geolocation Anomaly Trends
-- Tracks inbound transaction volume originating from designated high-risk territories.
-- -----------------------------------------------------------------------------
SELECT
    ip_country,
    COUNT(transaction_id) AS total_inflow_count,
    ROUND(SUM(amount), 2) AS total_inflow_volume_usd,
    COUNT(CASE WHEN is_quarantine = 1 THEN 1 END) AS quarantined_volume_count,
    ROUND(
        (COUNT(CASE WHEN is_quarantine = 1 THEN 1 END) * 100.0) / COUNT(transaction_id), 
        2
    ) AS quarantine_density_pct,
    ROUND(AVG(amount), 2) AS mean_inbound_amount
FROM delta.`data/delta/silver/transactions`
WHERE is_high_risk_country = 1 OR ip_country IN ('RU', 'NG', 'KP', 'IR', 'SY')
GROUP BY ip_country
ORDER BY total_inflow_volume_usd DESC;


-- -----------------------------------------------------------------------------
-- Query 11: Customer Lifetime Value (LTV) vs. Fraud Loss Exposure Quartiles
-- Groups customers into spend quartiles to analyze fraud concentration across customer segments.
-- -----------------------------------------------------------------------------
WITH customer_aggregates AS (
    SELECT
        customer_id,
        SUM(amount) AS total_customer_spend,
        COUNT(transaction_id) AS txn_count,
        SUM(CASE WHEN risk_score >= 80 THEN amount ELSE 0.0 END) AS quarantined_fraud_loss
    FROM delta.`data/delta/silver/transactions`
    GROUP BY customer_id
),
quartiled_customers AS (
    SELECT
        customer_id,
        total_customer_spend,
        txn_count,
        quarantined_fraud_loss,
        NTILE(4) OVER (ORDER BY total_customer_spend DESC) AS spend_quartile
    FROM customer_aggregates
)
SELECT
    spend_quartile,
    COUNT(customer_id) AS total_customers_in_quartile,
    ROUND(SUM(total_customer_spend), 2) AS quartile_gross_spend,
    ROUND(SUM(quarantined_fraud_loss), 2) AS quartile_fraud_exposure,
    ROUND(
        (SUM(quarantined_fraud_loss) * 100.0) / NULLIF(SUM(total_customer_spend), 0), 
        2
    ) AS fraud_to_spend_loss_ratio_pct
FROM quartiled_customers
GROUP BY spend_quartile
ORDER BY spend_quartile ASC;


-- -----------------------------------------------------------------------------
-- Query 12: Model Inference Probability vs. Deterministic Heuristic Risk Score Alignment
-- Compares Spark MLlib continuous probabilities with discrete rule scores.
-- -----------------------------------------------------------------------------
SELECT
    CASE
        WHEN risk_score >= 80 THEN 'Tier 1: Critical (80-100)'
        WHEN risk_score >= 60 THEN 'Tier 2: High (60-79)'
        WHEN risk_score >= 30 THEN 'Tier 3: Moderate (30-59)'
        ELSE 'Tier 4: Low (0-29)'
    END AS heuristic_risk_tier,
    COUNT(transaction_id) AS transaction_count,
    ROUND(AVG(risk_score), 1) AS mean_rule_score,
    ROUND(AVG(COALESCE(ml_fraud_probability, 0.0)), 3) AS mean_ml_predicted_prob,
    COUNT(CASE WHEN COALESCE(ml_fraud_probability, 0.0) >= 0.5 THEN 1 END) AS ml_flagged_count,
    ROUND(
        (COUNT(CASE WHEN COALESCE(ml_fraud_probability, 0.0) >= 0.5 THEN 1 END) * 100.0) / COUNT(transaction_id), 
        2
    ) AS ml_concordance_rate_pct
FROM delta.`data/delta/silver/transactions`
GROUP BY
    CASE
        WHEN risk_score >= 80 THEN 'Tier 1: Critical (80-100)'
        WHEN risk_score >= 60 THEN 'Tier 2: High (60-79)'
        WHEN risk_score >= 30 THEN 'Tier 3: Moderate (30-59)'
        ELSE 'Tier 4: Low (0-29)'
    END
ORDER BY mean_rule_score DESC;
