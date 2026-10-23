"""Velocity feature engineering engine for distributed fraud detection in PySpark.

Calculates multi-horizon rolling temporal aggregations (5m, 1h, 24h),
rapid-fire consecutive transaction bursts, and spending deviation metrics.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VelocityFeatureEngine:
    """Computes behavioral transaction velocity and spending divergence features."""

    @staticmethod
    def add_epoch_timestamp(df: DataFrame) -> DataFrame:
        """Adds Unix epoch timestamp in seconds for continuous range windowing."""
        if "event_time_epoch" not in df.columns:
            return df.withColumn("event_time_epoch", F.col("event_time").cast("long"))
        return df

    def compute_rolling_velocity_metrics(self, df: DataFrame) -> DataFrame:
        """Calculates rolling counts and monetary sums across 5m, 1h, and 24h horizons.

        Horizons:
        - 5 Minutes (300 seconds): Micro-burst detection (rapid card theft)
        - 1 Hour (3,600 seconds): Hourly accumulation detection
        - 24 Hours (86,400 seconds): Daily volume and daily cap breach detection

        Args:
            df: Ingested DataFrame containing customer_id, amount, and event_time.

        Returns:
            DataFrame with rolling velocity columns appended.
        """
        logger.info("Computing multi-horizon rolling velocity metrics (5m, 1h, 24h)...")
        df_epoch = self.add_epoch_timestamp(df)

        # Range windows based on epoch seconds
        w_5m = (
            Window.partitionBy("customer_id")
            .orderBy("event_time_epoch")
            .rangeBetween(-300, 0)
        )
        w_1h = (
            Window.partitionBy("customer_id")
            .orderBy("event_time_epoch")
            .rangeBetween(-3600, 0)
        )
        w_24h = (
            Window.partitionBy("customer_id")
            .orderBy("event_time_epoch")
            .rangeBetween(-86400, 0)
        )

        enriched = (
            df_epoch
            # 5-minute velocity
            .withColumn("txn_count_5m", F.count("transaction_id").over(w_5m))
            .withColumn("txn_sum_5m", F.round(F.sum("amount").over(w_5m), 2))
            # 1-hour velocity
            .withColumn("txn_count_1h", F.count("transaction_id").over(w_1h))
            .withColumn("txn_sum_1h", F.round(F.sum("amount").over(w_1h), 2))
            # 24-hour velocity
            .withColumn("txn_count_24h", F.count("transaction_id").over(w_24h))
            .withColumn("txn_sum_24h", F.round(F.sum("amount").over(w_24h), 2))
        )

        return enriched

    def compute_consecutive_deltas(self, df: DataFrame) -> DataFrame:
        """Calculates inter-transaction elapsed time and rapid consecutive burst counts.

        Computes:
        - prev_event_time: Timestamp of preceding customer transaction
        - prev_amount: Amount of preceding transaction
        - time_since_prev_txn_sec: Elapsed seconds between transactions
        - is_rapid_succession: Flag (1/0) if time_since_prev_txn_sec <= 60 seconds
        - rapid_txn_burst_5m: Count of rapid transactions in the rolling 5-minute window
        """
        logger.info("Calculating consecutive transaction deltas and rapid-fire intervals...")
        df_epoch = self.add_epoch_timestamp(df)

        w_cust_order = (
            Window.partitionBy("customer_id")
            .orderBy("event_time")
        )

        # Preceding transaction tracking
        with_lag = (
            df_epoch
            .withColumn("prev_event_time", F.lag("event_time", 1).over(w_cust_order))
            .withColumn("prev_amount", F.lag("amount", 1).over(w_cust_order))
            .withColumn(
                "prev_event_time_epoch",
                F.lag("event_time_epoch", 1).over(w_cust_order),
            )
            .withColumn(
                "time_since_prev_txn_sec",
                F.when(
                    F.col("prev_event_time_epoch").isNull(),
                    F.lit(999999),
                ).otherwise(
                    F.col("event_time_epoch") - F.col("prev_event_time_epoch")
                ),
            )
            .withColumn(
                "is_rapid_succession",
                F.when(F.col("time_since_prev_txn_sec") <= 60, F.lit(1)).otherwise(F.lit(0)),
            )
        )

        # Range window for rapid count
        w_5m = (
            Window.partitionBy("customer_id")
            .orderBy("event_time_epoch")
            .rangeBetween(-300, 0)
        )

        with_rapid_burst = with_lag.withColumn(
            "rapid_txn_burst_count",
            F.sum("is_rapid_succession").over(w_5m),
        )

        return with_rapid_burst

    def compute_amount_deviation(self, df: DataFrame) -> DataFrame:
        """Calculates spending divergence against baseline customer transaction amounts.

        Computes:
        - customer_baseline_amount: 30-day historical average or rolling average fallback
        - amount_to_avg_ratio: Ratio of current transaction amount to historical baseline
        - amount_diff_from_avg: Absolute numeric difference in currency units
        - is_amount_spike: Boolean indicator if ratio exceeds configured threshold (e.g. 5x)
        """
        logger.info("Computing spending deviation from historical customer baseline...")

        # Fallback baseline if historical_avg_amount_30d is null or 0
        w_cust_all = Window.partitionBy("customer_id")
        baseline_col = F.coalesce(
            F.when(F.col("historical_avg_amount_30d") > 0, F.col("historical_avg_amount_30d")),
            F.avg("amount").over(w_cust_all),
            F.lit(50.0),
        )

        return (
            df.withColumn("effective_baseline_amount", F.round(baseline_col, 2))
            .withColumn(
                "amount_to_avg_ratio",
                F.round(
                    F.col("amount")
                    / F.when(
                        F.col("effective_baseline_amount") <= 0, F.lit(1.0)
                    ).otherwise(F.col("effective_baseline_amount")),
                    4,
                ),
            )
            .withColumn(
                "amount_diff_from_avg",
                F.round(F.col("amount") - F.col("effective_baseline_amount"), 2),
            )
        )

    def transform(self, df: DataFrame) -> DataFrame:
        """Executes full velocity transformation pipeline."""
        logger.info("Executing comprehensive Velocity Feature transformation pipeline...")
        step1 = self.compute_rolling_velocity_metrics(df)
        step2 = self.compute_consecutive_deltas(step1)
        step3 = self.compute_amount_deviation(step2)
        logger.info("Velocity Feature transformation complete.")
        return step3
