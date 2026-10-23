"""Master pipeline orchestrator for pyspark-fintech-fraud-engine.

Provides a robust CLI interface to run the full Medallion lakehouse lifecycle:
Ingestion -> Velocity & Geo Feature Engineering -> Deterministic Rules -> Spark MLlib -> Delta ACID Merge.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import config
from src.extract.ingestion import TransactionIngestionEngine
from src.features.geo_features import GeoFeatureEngine
from src.features.velocity_features import VelocityFeatureEngine
from src.load.delta_loader import DeltaLakeLoader
from src.ml.fraud_model import FraudMLPipeline
from src.rules.rule_engine import FraudRuleEngine
from src.utils.logger import get_logger
from src.utils.spark_session import get_spark_session, stop_spark_session

logger = get_logger("fintech_fraud_pipeline")


class FraudDetectionPipeline:
    """Orchestrates end-to-end data processing, feature calculation, scoring, and persistence."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.ingestion = TransactionIngestionEngine(spark)
        self.velocity_engine = VelocityFeatureEngine()
        self.geo_engine = GeoFeatureEngine()
        self.rule_engine = FraudRuleEngine()
        self.ml_pipeline = FraudMLPipeline()
        self.loader = DeltaLakeLoader(spark)

    def run_batch(
        self,
        source_path: str,
        run_ml: bool = False,
        classifier_type: str = "gbt",
        optimize: bool = False,
    ) -> None:
        """Executes full batch lifecycle from raw file to Gold marts and quarantine."""
        start_time = time.time()
        logger.info("=================================================================")
        logger.info("Starting FinTech Fraud Detection Batch Pipeline")
        logger.info("Source: %s | ML Enabled: %s | Classifier: %s", source_path, run_ml, classifier_type)
        logger.info("=================================================================")

        # Step 1: Ingestion & Bronze Write
        logger.info(">>> Stage 1: Ingestion & Bronze Storage")
        if source_path.endswith(".csv"):
            raw_df = self.ingestion.read_csv(source_path)
        elif source_path.endswith(".parquet"):
            raw_df = self.ingestion.read_parquet(source_path)
        else:
            raise ValueError(f"Unsupported file format for source_path: {source_path}")

        self.ingestion.write_to_bronze(raw_df)

        # Step 2: Feature Engineering (Velocity + Geo)
        logger.info(">>> Stage 2: Feature Engineering")
        velocity_df = self.velocity_engine.transform(raw_df)
        geo_df = self.geo_engine.transform(velocity_df)

        # Step 3: Deterministic Rule Scoring
        logger.info(">>> Stage 3: Deterministic Rule Engine Scoring")
        scored_df = self.rule_engine.evaluate_rules(geo_df)

        # Step 4: Machine Learning Inference (Optional)
        final_df = scored_df
        if run_ml:
            logger.info(">>> Stage 4: Spark MLlib Model Training & Scoring")
            _, metrics = self.ml_pipeline.train(
                scored_df, label_col="is_fraud_label", classifier_type=classifier_type
            )
            logger.info("MLlib Training Metrics: %s", metrics)
            final_df = self.ml_pipeline.predict(scored_df)

        # Step 5: Medallion Loading (Silver Upsert, Quarantine Routing, Gold Marts)
        logger.info(">>> Stage 5: Delta Lake ACID Upsert & Alert Mart Publishing")
        self.loader.upsert_silver(final_df)
        quarantine_count = self.loader.route_quarantine_alerts(
            final_df, score_threshold=config.rules.quarantine_risk_score_threshold
        )
        self.loader.publish_gold_marts(final_df)

        # Step 6: Delta Table Maintenance (Compaction & Z-ORDER)
        if optimize:
            logger.info(">>> Stage 6: Delta Lake Maintenance & Z-ORDER Optimization")
            self.loader.optimize_and_vacuum(
                config.delta.silver_path,
                zorder_cols=["customer_id", "event_time"],
            )

        elapsed = round(time.time() - start_time, 2)
        logger.info("=================================================================")
        logger.info(
            "Batch Pipeline Execution Finished Successfully in %s seconds!", elapsed
        )
        logger.info("Quarantined High-Risk Transactions: %d", quarantine_count)
        logger.info("=================================================================")


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="pyspark-fintech-fraud-engine - Large-Scale Fraud Detection Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "streaming", "optimize-only"],
        default="batch",
        help="Pipeline operational mode",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "data" / "synthetic_transactions.csv"
        ),
        help="Input path to raw transaction file or directory",
    )
    parser.add_argument(
        "--run-ml",
        action="store_true",
        help="Train Spark MLlib model and compute probability predictions",
    )
    parser.add_argument(
        "--classifier-type",
        choices=["gbt", "rf"],
        default="gbt",
        help="ML classifier type: gbt (Gradient-Boosted Trees) or rf (Random Forest)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Execute Delta OPTIMIZE and Z-ORDER after load",
    )
    parser.add_argument(
        "--optimize-only",
        action="store_true",
        help="Skip pipeline processing and only optimize existing Delta tables",
    )

    return parser.parse_args()


def main() -> None:
    """Application entrypoint."""
    args = parse_arguments()
    spark = get_spark_session()

    try:
        if args.optimize_only:
            logger.info("Running Delta Optimization Only...")
            loader = DeltaLakeLoader(spark)
            loader.optimize_and_vacuum(
                config.delta.silver_path,
                zorder_cols=["customer_id", "event_time"],
            )
            return

        pipeline = FraudDetectionPipeline(spark)
        pipeline.run_batch(
            source_path=args.source_path,
            run_ml=args.run_ml,
            classifier_type=args.classifier_type,
            optimize=args.optimize,
        )
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()
