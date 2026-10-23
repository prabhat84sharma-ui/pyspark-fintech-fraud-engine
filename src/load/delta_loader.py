"""Delta Lake loader and Lakehouse lifecycle management for FinTech Fraud Engine.

Implements ACID MERGE upserts, quarantine alert routing, Gold aggregations,
and automated Delta maintenance (compaction, Z-ORDER, and VACUUM).
"""

from __future__ import annotations

import os
from typing import Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import config, DeltaPathConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeltaLakeLoader:
    """Manages transactional ACID writes, Medallion merges, and table optimizations."""

    def __init__(self, spark: SparkSession, delta_paths: Optional[DeltaPathConfig] = None):
        self.spark = spark
        self.paths = delta_paths or config.delta

    def upsert_silver(
        self,
        df: DataFrame,
        silver_path: Optional[str] = None,
        partition_col: str = "event_date",
    ) -> None:
        """Upserts scored transactions into Silver Delta table by transaction_id.

        If the table does not exist, performs initial partitioned write.
        If the table exists, executes an ACID MERGE to ensure idempotent ingestion.
        """
        dest_path = silver_path or self.paths.silver_path
        logger.info("Upserting transactions into Silver Delta Table at: %s", dest_path)

        from delta.tables import DeltaTable

        if DeltaTable.isDeltaTable(self.spark, dest_path):
            logger.info("Existing Delta table detected at %s. Executing MERGE operation...", dest_path)
            delta_target = DeltaTable.forPath(self.spark, dest_path)

            (
                delta_target.alias("target")
                .merge(
                    df.alias("source"),
                    "target.transaction_id = source.transaction_id",
                )
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
            logger.info("ACID MERGE completed successfully into Silver table.")
        else:
            logger.info("No Delta table at %s. Initializing new Silver Delta table...", dest_path)
            (
                df.write.format("delta")
                .mode("overwrite")
                .partitionBy(partition_col)
                .option("mergeSchema", "true")
                .save(dest_path)
            )
            logger.info("Silver Delta table initialized successfully.")

    def route_quarantine_alerts(
        self,
        df: DataFrame,
        quarantine_path: Optional[str] = None,
        score_threshold: int = 80,
    ) -> int:
        """Filters high-risk transactions (risk_score >= 80) into the Quarantine Alert Mart.

        Returns:
            Count of quarantined transactions.
        """
        dest_path = quarantine_path or self.paths.quarantine_alerts_path
        logger.info("Routing transactions with risk_score >= %d to Quarantine Mart: %s", score_threshold, dest_path)

        alerts_df = df.filter(F.col("risk_score") >= score_threshold)
        alert_count = alerts_df.count()

        if alert_count == 0:
            logger.info("Zero transactions breached quarantine threshold (%d).", score_threshold)
            return 0

        logger.warning("Routing %d high-risk transactions to Quarantine Alert Mart!", alert_count)

        from delta.tables import DeltaTable

        if DeltaTable.isDeltaTable(self.spark, dest_path):
            delta_target = DeltaTable.forPath(self.spark, dest_path)
            (
                delta_target.alias("target")
                .merge(
                    alerts_df.alias("source"),
                    "target.transaction_id = source.transaction_id",
                )
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
        else:
            (
                alerts_df.write.format("delta")
                .mode("append")
                .partitionBy("event_date")
                .option("mergeSchema", "true")
                .save(dest_path)
            )

        logger.info("Quarantine Mart updated with %d active alerts.", alert_count)
        return alert_count

    def publish_gold_marts(self, silver_df: DataFrame) -> None:
        """Computes and updates Gold Lakehouse summary and feature store marts."""
        logger.info("Publishing Gold Feature Store and Analytical Marts...")

        # 1. Gold Customer Risk Profile Feature Store
        customer_gold = (
            silver_df.groupBy("customer_id")
            .agg(
                F.count("transaction_id").alias("total_lifetime_txns"),
                F.round(F.sum("amount"), 2).alias("total_spend_amount"),
                F.round(F.avg("amount"), 2).alias("avg_txn_amount"),
                F.max("risk_score").alias("max_historical_risk_score"),
                F.sum(F.when(F.col("risk_score") >= 60, 1).otherwise(0)).alias("high_risk_txn_count"),
                F.max("event_time").alias("last_active_timestamp"),
                F.round(F.avg("risk_score"), 1).alias("mean_risk_score"),
            )
            .withColumn("profile_updated_at", F.current_timestamp())
        )

        customer_path = self.paths.gold_customer_features_path
        logger.info("Writing Customer Risk Feature Store to: %s", customer_path)
        customer_gold.write.format("delta").mode("overwrite").save(customer_path)

        # 2. Gold Merchant & Channel Fraud Exposure Mart
        merchant_gold = (
            silver_df.groupBy("merchant_category", "channel")
            .agg(
                F.count("transaction_id").alias("total_txns"),
                F.round(F.sum("amount"), 2).alias("total_volume"),
                F.sum(F.when(F.col("risk_score") >= 60, 1).otherwise(0)).alias("flagged_txns"),
                F.round(
                    F.sum(F.when(F.col("risk_score") >= 60, F.col("amount")).otherwise(0.0)),
                    2,
                ).alias("flagged_fraud_volume"),
                F.round(
                    (F.sum(F.when(F.col("risk_score") >= 60, 1).otherwise(0)) / F.count("transaction_id")) * 100.0,
                    2,
                ).alias("fraud_rate_pct"),
            )
            .withColumn("mart_updated_at", F.current_timestamp())
        )

        merchant_path = self.paths.gold_merchant_metrics_path
        logger.info("Writing Merchant Exposure Mart to: %s", merchant_path)
        merchant_gold.write.format("delta").mode("overwrite").save(merchant_path)

        logger.info("Gold Marts successfully published.")

    def optimize_and_vacuum(
        self, table_path: str, zorder_cols: Optional[list[str]] = None, vacuum_retention_hours: Optional[int] = None
    ) -> None:
        """Executes Delta compaction, Z-ORDER clustering, and snapshot vacuuming."""
        from delta.tables import DeltaTable

        if not DeltaTable.isDeltaTable(self.spark, table_path):
            logger.warning("Path %s is not a valid Delta table; skipping optimization.", table_path)
            return

        dt = DeltaTable.forPath(self.spark, table_path)
        logger.info("Optimizing Delta table at: %s", table_path)

        try:
            if zorder_cols:
                logger.info("Executing OPTIMIZE with Z-ORDER BY %s...", zorder_cols)
                dt.optimize().executeZOrderBy(*zorder_cols)
            else:
                dt.optimize().executeCompaction()
            logger.info("Optimization complete for %s.", table_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("OPTIMIZE operation skipped or unsupported in environment: %s", exc)

        if vacuum_retention_hours is not None:
            try:
                logger.info("Running VACUUM (retention: %d hours)...", vacuum_retention_hours)
                dt.vacuum(vacuum_retention_hours)
                logger.info("VACUUM complete for %s.", table_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("VACUUM operation skipped: %s", exc)
