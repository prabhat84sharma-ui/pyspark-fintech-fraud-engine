"""Unit tests for Velocity and Geo-Spatial feature engineering pipelines."""

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
    StringType,
    StructField,
    StructType,
)

from src.features.geo_features import GeoFeatureEngine
from src.features.velocity_features import VelocityFeatureEngine


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for transaction velocity and geo-spatial analytics."""

    spark: SparkSession

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("pyspark-fraud-test-features")
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

    def test_velocity_rolling_metrics(self):
        """Verifies that rolling counts and sums compute correctly over temporal range windows."""
        schema = StructType(
            [
                StructField("transaction_id", StringType(), False),
                StructField("customer_id", StringType(), False),
                StructField("event_time", StringType(), False),
                StructField("amount", DoubleType(), False),
                StructField("historical_avg_amount_30d", DoubleType(), True),
            ]
        )

        data = [
            ("T1", "C1", "2026-10-23 10:00:00", 100.0, 50.0),
            ("T2", "C1", "2026-10-23 10:02:00", 200.0, 50.0),  # within 5m
            ("T3", "C1", "2026-10-23 10:04:00", 300.0, 50.0),  # within 5m
            ("T4", "C1", "2026-10-23 10:45:00", 50.0, 50.0),   # outside 5m, within 1h
            ("T5", "C1", "2026-10-23 14:00:00", 80.0, 50.0),   # outside 1h, within 24h
        ]

        raw_df = self.spark.createDataFrame(data, schema)
        engine = VelocityFeatureEngine()
        transformed = engine.transform(raw_df).collect()

        t_map = {row["transaction_id"]: row for row in transformed}

        # T1: first transaction
        self.assertEqual(t_map["T1"]["txn_count_5m"], 1)
        self.assertEqual(t_map["T1"]["txn_sum_5m"], 100.0)
        self.assertEqual(t_map["T1"]["txn_count_1h"], 1)

        # T2: 2 minutes after T1
        self.assertEqual(t_map["T2"]["txn_count_5m"], 2)
        self.assertEqual(t_map["T2"]["txn_sum_5m"], 300.0)
        self.assertEqual(t_map["T2"]["time_since_prev_txn_sec"], 120)

        # T3: 4 minutes after T1, 2 minutes after T2
        self.assertEqual(t_map["T3"]["txn_count_5m"], 3)
        self.assertEqual(t_map["T3"]["txn_sum_5m"], 600.0)
        self.assertEqual(t_map["T3"]["txn_count_1h"], 3)
        self.assertEqual(t_map["T3"]["txn_sum_1h"], 600.0)

        # T4: 45 min after T1 (outside 5m, within 1h)
        self.assertEqual(t_map["T4"]["txn_count_5m"], 1)
        self.assertEqual(t_map["T4"]["txn_sum_5m"], 50.0)
        self.assertEqual(t_map["T4"]["txn_count_1h"], 4)
        self.assertEqual(t_map["T4"]["txn_sum_1h"], 650.0)

        # T5: 4h after T1 (outside 1h, within 24h)
        self.assertEqual(t_map["T5"]["txn_count_5m"], 1)
        self.assertEqual(t_map["T5"]["txn_count_1h"], 1)
        self.assertEqual(t_map["T5"]["txn_count_24h"], 5)
        self.assertEqual(t_map["T5"]["txn_sum_24h"], 730.0)

    def test_velocity_amount_deviation(self):
        """Verifies that spending deviation and multiplier ratios are accurately derived."""
        schema = StructType(
            [
                StructField("transaction_id", StringType(), False),
                StructField("customer_id", StringType(), False),
                StructField("event_time", StringType(), False),
                StructField("amount", DoubleType(), False),
                StructField("historical_avg_amount_30d", DoubleType(), True),
            ]
        )

        data = [
            ("T1", "C2", "2026-10-23 12:00:00", 500.0, 100.0),
            ("T2", "C2", "2026-10-23 12:05:00", 50.0, 100.0),
        ]

        raw_df = self.spark.createDataFrame(data, schema)
        engine = VelocityFeatureEngine()
        transformed = engine.compute_amount_deviation(raw_df).collect()

        t_map = {row["transaction_id"]: row for row in transformed}
        self.assertEqual(t_map["T1"]["amount_to_avg_ratio"], 5.0)
        self.assertEqual(t_map["T1"]["amount_diff_from_avg"], 400.0)
        self.assertEqual(t_map["T2"]["amount_to_avg_ratio"], 0.5)
        self.assertEqual(t_map["T2"]["amount_diff_from_avg"], -50.0)

    def test_geo_haversine_and_impossible_travel(self):
        """Tests Haversine distance, speed calculation, and impossible travel flag detection."""
        schema = StructType(
            [
                StructField("transaction_id", StringType(), False),
                StructField("customer_id", StringType(), False),
                StructField("event_time", StringType(), False),
                StructField("latitude", DoubleType(), True),
                StructField("longitude", DoubleType(), True),
                StructField("ip_country", StringType(), True),
            ]
        )

        # NYC to London (approx 5570 km) within 45 minutes -> ~7400 km/h (Impossible travel)
        data = [
            ("T1", "C3", "2026-10-23 14:00:00", 40.7128, -74.0060, "US"),
            ("T2", "C3", "2026-10-23 14:45:00", 51.5074, -0.1278, "GB"),
        ]

        raw_df = self.spark.createDataFrame(data, schema)
        engine = GeoFeatureEngine()
        transformed = engine.transform(raw_df).collect()

        t_map = {row["transaction_id"]: row for row in transformed}

        self.assertEqual(t_map["T1"]["haversine_distance_km"], 0.0)
        self.assertEqual(t_map["T1"]["is_impossible_travel"], 0)

        # T2 should be > 5500 km and flagged as impossible travel (> 800 km/h)
        self.assertGreater(t_map["T2"]["haversine_distance_km"], 5500.0)
        self.assertGreater(t_map["T2"]["travel_speed_kmh"], 7000.0)
        self.assertEqual(t_map["T2"]["is_impossible_travel"], 1)
        self.assertEqual(t_map["T2"]["is_cross_border_txn"], 1)


if __name__ == "__main__":
    unittest.main()
