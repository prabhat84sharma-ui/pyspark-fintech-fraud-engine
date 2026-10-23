"""Deterministic fraud rules engine and composite risk scoring for FinTech payments.

Evaluates multi-vector heuristic rules against transactions, assigns weighted risk scores (0-100),
generates auditable reason codes, and tags transactions for automated quarantine routing.
"""

from __future__ import annotations

from typing import Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import config, FraudRuleThresholds
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudRuleEngine:
    """Evaluates business and security rules to generate deterministic risk scores."""

    def __init__(self, thresholds: Optional[FraudRuleThresholds] = None):
        self.thresholds = thresholds or config.rules

    def evaluate_rules(self, df: DataFrame) -> DataFrame:
        """Applies deterministic fraud rules and outputs reason codes and score components.

        Rules Evaluated:
        1. GEO_IMPOSSIBLE_TRAVEL (45 pts): Velocity between consecutive transactions exceeds human physical travel.
        2. AMOUNT_SPIKE_5X (30 pts): Transaction amount exceeds 5x historical 30-day baseline and absolute threshold.
        3. VELOCITY_BURST_5M (25 pts): 4 or more transactions executed within a 5-minute rolling window.
        4. HIGH_RISK_JURISDICTION (20 pts): Transaction originated from sanctioned or high-risk IP country.
        5. SUSPICIOUS_NOCTURNAL_SPIKE (20 pts): Elevated charge initiated during 01:00-04:59 local nocturnal window.
        6. RAPID_CARD_TESTING (25 pts): Micro-charges in rapid succession (< 60s) indicating card brute-forcing.

        Args:
            df: Enriched DataFrame containing velocity and geo features.

        Returns:
            DataFrame with individual rule flags, risk score, and reason codes array.
        """
        logger.info("Evaluating deterministic fraud detection rules...")

        # Extract transaction hour of day
        with_hour = df.withColumn("event_hour", F.hour(F.col("event_time")))

        # Rule 1: Impossible Travel
        r1_cond = (
            (F.col("is_impossible_travel") == 1)
            | (
                (F.col("travel_speed_kmh") > config.geo.max_human_travel_speed_kmh)
                & (F.col("haversine_distance_km") >= config.geo.min_distance_for_speed_check_km)
            )
        )
        r1_pts = F.when(r1_cond, F.lit(45)).otherwise(F.lit(0))
        r1_code = F.when(r1_cond, F.lit("GEO_IMPOSSIBLE_TRAVEL"))

        # Rule 2: Amount Spike (> 5x historical mean and > $500)
        r2_cond = (
            (F.col("amount_to_avg_ratio") >= self.thresholds.amount_spike_multiplier)
            & (F.col("amount") >= self.thresholds.minimum_spike_absolute_amount)
        )
        r2_pts = F.when(r2_cond, F.lit(30)).otherwise(F.lit(0))
        r2_code = F.when(r2_cond, F.lit("AMOUNT_SPIKE_5X"))

        # Rule 3: Velocity Burst (>= 4 transactions in 5 min or >= 8 in 1h)
        r3_cond = (
            (F.col("txn_count_5m") >= self.thresholds.velocity_5m_txn_count_threshold)
            | (F.col("txn_count_1h") >= self.thresholds.velocity_1h_txn_count_threshold)
        )
        r3_pts = F.when(r3_cond, F.lit(25)).otherwise(F.lit(0))
        r3_code = F.when(r3_cond, F.lit("VELOCITY_BURST_5M"))

        # Rule 4: High-Risk Jurisdiction
        r4_cond = (
            (F.col("is_high_risk_country") == 1)
            | F.col("ip_country").isin(config.geo.high_risk_country_codes)
        )
        r4_pts = F.when(r4_cond, F.lit(20)).otherwise(F.lit(0))
        r4_code = F.when(r4_cond, F.lit("HIGH_RISK_JURISDICTION"))

        # Rule 5: Suspicious Nocturnal Spike (hours 1, 2, 3, 4 with amount >= nocturnal threshold)
        r5_cond = (
            F.col("event_hour").isin(list(self.thresholds.nocturnal_hours))
            & (F.col("amount") >= self.thresholds.nocturnal_amount_threshold)
        )
        r5_pts = F.when(r5_cond, F.lit(20)).otherwise(F.lit(0))
        r5_code = F.when(r5_cond, F.lit("SUSPICIOUS_NOCTURNAL_SPIKE"))

        # Rule 6: Rapid Card Testing (< $5 within <= 60s burst)
        r6_cond = (
            (F.col("amount") <= self.thresholds.card_testing_max_amount)
            & (F.col("time_since_prev_txn_sec") <= self.thresholds.card_testing_time_window_sec)
            & (F.col("rapid_txn_burst_count") >= 2)
        )
        r6_pts = F.when(r6_cond, F.lit(25)).otherwise(F.lit(0))
        r6_code = F.when(r6_cond, F.lit("RAPID_CARD_TESTING"))

        # Raw composite sum capped at 100
        raw_score = r1_pts + r2_pts + r3_pts + r4_pts + r5_pts + r6_pts
        capped_score = F.least(F.lit(100), raw_score)

        # Build clean reason codes array
        all_codes = F.array(r1_code, r2_code, r3_code, r4_code, r5_code, r6_code)
        clean_codes = F.filter(all_codes, lambda x: x.isNotNull())

        # Decision category
        decision_col = (
            F.when(capped_score >= self.thresholds.quarantine_risk_score_threshold, F.lit("QUARANTINE_ALERT"))
            .when(capped_score >= self.thresholds.high_risk_score_threshold, F.lit("REVIEW_REQUIRED"))
            .when(capped_score >= 30, F.lit("MONITOR"))
            .otherwise(F.lit("APPROVE"))
        )

        scored_df = (
            with_hour
            .withColumn("rule_impossible_travel", F.when(r1_cond, F.lit(1)).otherwise(F.lit(0)))
            .withColumn("rule_amount_spike", F.when(r2_cond, F.lit(1)).otherwise(F.lit(0)))
            .withColumn("rule_velocity_burst", F.when(r3_cond, F.lit(1)).otherwise(F.lit(0)))
            .withColumn("rule_high_risk_geo", F.when(r4_cond, F.lit(1)).otherwise(F.lit(0)))
            .withColumn("rule_nocturnal_spike", F.when(r5_cond, F.lit(1)).otherwise(F.lit(0)))
            .withColumn("rule_card_testing", F.when(r6_cond, F.lit(1)).otherwise(F.lit(0)))
            .withColumn("risk_score", capped_score.cast("integer"))
            .withColumn("flagged_rule_codes", clean_codes)
            .withColumn("fraud_decision", decision_col)
            .withColumn("is_quarantine", F.when(capped_score >= self.thresholds.quarantine_risk_score_threshold, F.lit(1)).otherwise(F.lit(0)))
        )

        return scored_df
