"""Production SparkSession factory with Delta Lake 3.x integration and AQE tuning."""

from __future__ import annotations

import os
from typing import Optional
from pyspark.sql import SparkSession

from src.config import config, SparkTuningConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_spark_session(
    tuning_config: Optional[SparkTuningConfig] = None,
    extra_configs: Optional[dict[str, str]] = None,
) -> SparkSession:
    """Builds and returns a tuned SparkSession with Delta Lake 3.x support.

    Configures:
    - Delta Lake catalog extensions and SQL parser extensions
    - Adaptive Query Execution (AQE) with partition coalescing and skew-join handling
    - Dynamic partition pruning
    - Optimized writes and auto-compaction for Delta Lake tables
    - Memory overhead and shuffle partition bounds

    Args:
        tuning_config: Optional configuration override. Defaults to global config.spark.
        extra_configs: Additional Spark configuration key-value pairs.

    Returns:
        Configured SparkSession instance.
    """
    cfg = tuning_config or config.spark

    logger.info(
        "Initializing SparkSession: app_name=%s, master=%s, shuffle_partitions=%d",
        cfg.app_name,
        cfg.master,
        cfg.shuffle_partitions,
    )

    builder = (
        SparkSession.builder.appName(cfg.app_name)
        .master(cfg.master)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Adaptive Query Execution (AQE)
        .config("spark.sql.adaptive.enabled", str(cfg.enable_aqe).lower())
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
        # Dynamic Partition Pruning
        .config("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
        # Shuffle tuning
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.default.parallelism", str(cfg.shuffle_partitions))
        # Delta Lake write optimization
        .config(
            "spark.databricks.delta.optimizeWrite.enabled",
            str(cfg.enable_delta_optimize_write).lower(),
        )
        .config(
            "spark.databricks.delta.autoCompact.enabled",
            str(cfg.enable_delta_auto_compact).lower(),
        )
        # Timezone & Serialization
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # Driver and executor memory defaults
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.executor.memory", cfg.executor_memory)
    )

    # Apply any custom config overrides
    if extra_configs:
        for key, val in extra_configs.items():
            builder = builder.config(key, val)

    # Attempt delta-spark configure helper if installed
    try:
        from delta import configure_spark_with_delta_pip

        spark = configure_spark_with_delta_pip(builder).getOrCreate()
    except ImportError:
        logger.warning(
            "delta-spark pip helper not available; constructing SparkSession with raw builder"
        )
        spark = builder.getOrCreate()

    # Suppress verbose Spark internal logs in driver console
    spark.sparkContext.setLogLevel("WARN")

    logger.info("SparkSession successfully initialized (version: %s)", spark.version)
    return spark


def stop_spark_session(spark: Optional[SparkSession] = None) -> None:
    """Safely terminates the active SparkSession."""
    if spark is not None:
        try:
            spark.stop()
            logger.info("SparkSession stopped cleanly.")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Error while stopping SparkSession: %s", exc)
