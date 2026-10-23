"""Unit tests for Deterministic Fraud Rules and Composite Risk Scoring."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.rules.rule_engine import FraudRuleEngine


class TestFraudRules(unittest.TestCase):
    """Test suite for deterministic fraud rules and composite scoring logic."""

    spark: SparkSession

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("pyspark-fraud-test-rules")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        if cls.spark:
            cls.spark.stop()

    def test_rule_impossible_travel(self):
        """Verifies that impossible travel rule awards 45 points and flags the reason code."""
        schema = StructType(
            [
                StructField("transaction_id", StringType(), False),
                StructField("event_time", StringType(), False),
                StructField("amount", DoubleType(), False),
                StructField("amount_to_avg_ratio", DoubleType(), False),
                StructField("txn_count_5m", IntegerType(), False),
                StructField("txn_count_1h", IntegerType(), False),
                StructField("time_since_prev_txn_sec", IntegerType(), False),
                StructField("rapid_txn_burst_count", IntegerType(), False),
                StructField("haversine_distance_km", DoubleType(), False),
                StructField("travel_speed_kmh", DoubleType(), False),
                StructField("is_impossible_travel", IntegerType(), False),
                StructField("is_high_risk_country", IntegerType(), False),
                StructField("ip_country", StringType(), False),
            ]
        )

        data = [
            ("T1", "2026-10-23 15:00:00", 100.0, 1.0, 1, 1, 999999, 0, 5600.0, 7400.0, 1, 0, "GB"),
            ("T2", "2026-10-23 15:30:00", 50.0, 1.0, 1, 1, 1800, 0, 2.0, 4.0, 0, 0, "US"),
        ]

        df = self.spark.createDataFrame(data, schema)
        engine = FraudRuleEngine()
        scored = engine.evaluate_rules(df).collect()

        t_map = {row["transaction_id"]: row for row in scored}

        self.assertEqual(t_map["T1"]["rule_impossible_travel"], 1)
        self.assertIn("GEO_IMPOSSIBLE_TRAVEL", t_map["T1"]["flagged_rule_codes"])
        self.assertEqual(t_map["T1"]["risk_score"], 45)

        self.assertEqual(t_map["T2"]["rule_impossible_travel"], 0)
        self.assertNotIn("GEO_IMPOSSIBLE_TRAVEL", t_map["T2"]["flagged_rule_codes"])
        self.assertEqual(t_map["T2"]["risk_score"], 0)
        self.assertEqual(t_map["T2"]["fraud_decision"], "APPROVE")

    def test_rule_amount_spike_and_velocity_burst(self):
        """Verifies that amount spike and velocity burst flags combine into composite scores."""
        schema = StructType(
            [
                StructField("transaction_id", StringType(), False),
                StructField("event_time", StringType(), False),
                StructField("amount", DoubleType(), False),
                StructField("amount_to_avg_ratio", DoubleType(), False),
                StructField("txn_count_5m", IntegerType(), False),
                StructField("txn_count_1h", IntegerType(), False),
                StructField("time_since_prev_txn_sec", IntegerType(), False),
                StructField("rapid_txn_burst_count", IntegerType(), False),
                StructField("haversine_distance_km", DoubleType(), False),
                StructField("travel_speed_kmh", DoubleType(), False),
                StructField("is_impossible_travel", IntegerType(), False),
                StructField("is_high_risk_country", IntegerType(), False),
                StructField("ip_country", StringType(), False),
            ]
        )

        data = [
            # T1: 5.5x amount spike ($2200), velocity burst (5 txns in 5 min) -> 30 + 25 = 55
            ("T1", "2026-10-23 16:00:00", 2200.0, 5.5, 5, 5, 20, 3, 0.0, 0.0, 0, 0, "US"),
            # T2: Normal amount ($45), normal count (1) -> 0 pts
            ("T2", "2026-10-23 16:10:00", 45.0, 1.0, 1, 1, 600, 0, 0.0, 0.0, 0, 0, "US"),
        ]

        df = self.spark.createDataFrame(data, schema)
        engine = FraudRuleEngine()
        scored = engine.evaluate_rules(df).collect()

        t_map = {row["transaction_id"]: row for row in scored}

        self.assertEqual(t_map["T1"]["rule_amount_spike"], 1)
        self.assertEqual(t_map["T1"]["rule_velocity_burst"], 1)
        self.assertIn("AMOUNT_SPIKE_5X", t_map["T1"]["flagged_rule_codes"])
        self.assertIn("VELOCITY_BURST_5M", t_map["T1"]["flagged_rule_codes"])
        self.assertEqual(t_map["T1"]["risk_score"], 55)
        self.assertEqual(t_map["T1"]["fraud_decision"], "MONITOR")

        self.assertEqual(t_map["T2"]["risk_score"], 0)
        self.assertEqual(t_map["T2"]["fraud_decision"], "APPROVE")

    def test_quarantine_score_capping(self):
        """Verifies that multiple severe triggers sum and properly cap at 100 points, triggering quarantine."""
        schema = StructType(
            [
                StructField("transaction_id", StringType(), False),
                StructField("event_time", StringType(), False),
                StructField("amount", DoubleType(), False),
                StructField("amount_to_avg_ratio", DoubleType(), False),
                StructField("txn_count_5m", IntegerType(), False),
                StructField("txn_count_1h", IntegerType(), False),
                StructField("time_since_prev_txn_sec", IntegerType(), False),
                StructField("rapid_txn_burst_count", IntegerType(), False),
                StructField("haversine_distance_km", DoubleType(), False),
                StructField("travel_speed_kmh", DoubleType(), False),
                StructField("is_impossible_travel", IntegerType(), False),
                StructField("is_high_risk_country", IntegerType(), False),
                StructField("ip_country", StringType(), False),
            ]
        )

        data = [
            # T1: Impossible travel (45) + Amount spike (30) + Velocity burst (25) + High risk country (20) = 120 -> Capped at 100
            ("T1", "2026-10-23 03:00:00", 3500.0, 6.0, 5, 6, 25, 3, 5600.0, 8000.0, 1, 1, "RU"),
        ]

        df = self.spark.createDataFrame(data, schema)
        engine = FraudRuleEngine()
        scored = engine.evaluate_rules(df).collect()

        record = scored[0]
        self.assertEqual(record["risk_score"], 100)
        self.assertEqual(record["fraud_decision"], "QUARANTINE_ALERT")
        self.assertEqual(record["is_quarantine"], 1)
        self.assertGreaterEqual(len(record["flagged_rule_codes"]), 4)


if __name__ == "__main__":
    unittest.main()
