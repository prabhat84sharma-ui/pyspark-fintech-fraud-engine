"""Configuration module for pyspark-fintech-fraud-engine.

Provides centralized, production-grade configuration definitions using dataclasses
and environment variable overrides for Delta Lake paths, Spark tuning parameters,
deterministic fraud heuristics, and geo-spatial velocity boundaries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set


@dataclass(frozen=True)
class DeltaPathConfig:
    """Paths for Medallion architecture Delta tables and checkpoints."""

    base_dir: str = field(
        default_factory=lambda: os.getenv(
            "DELTA_BASE_DIR",
            str(Path(__file__).resolve().parent.parent / "data" / "delta")
        )
    )

    @property
    def bronze_path(self) -> str:
        """Raw multi-source ingested transactions with audit metadata."""
        return os.path.join(self.base_dir, "bronze", "transactions")

    @property
    def silver_path(self) -> str:
        """Enriched, deduplicated transactions with calculated features and scores."""
        return os.path.join(self.base_dir, "silver", "transactions")

    @property
    def gold_customer_features_path(self) -> str:
        """Gold feature store for customer risk and 30-day velocity aggregations."""
        return os.path.join(self.base_dir, "gold", "customer_features")

    @property
    def gold_merchant_metrics_path(self) -> str:
        """Gold aggregated fraud metrics by merchant category and channel."""
        return os.path.join(self.base_dir, "gold", "merchant_metrics")

    @property
    def quarantine_alerts_path(self) -> str:
        """High-risk quarantine table for immediate SOC/Fraud Analyst review."""
        return os.path.join(self.base_dir, "quarantine", "fraud_alerts")

    @property
    def checkpoint_path(self) -> str:
        """Checkpoint directory for Structured Streaming state stores."""
        return os.path.join(self.base_dir, "checkpoints")


@dataclass(frozen=True)
class SparkTuningConfig:
    """Spark performance and resource allocation configuration settings."""

    app_name: str = field(
        default_factory=lambda: os.getenv("SPARK_APP_NAME", "pyspark-fintech-fraud-engine")
    )
    master: str = field(
        default_factory=lambda: os.getenv("SPARK_MASTER", "local[*]")
    )
    shuffle_partitions: int = field(
        default_factory=lambda: int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
    )
    driver_memory: str = field(
        default_factory=lambda: os.getenv("SPARK_DRIVER_MEMORY", "4g")
    )
    executor_memory: str = field(
        default_factory=lambda: os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
    )
    enable_aqe: bool = field(
        default_factory=lambda: os.getenv("SPARK_ENABLE_AQE", "true").lower() == "true"
    )
    enable_dynamic_allocation: bool = field(
        default_factory=lambda: os.getenv("SPARK_DYNAMIC_ALLOCATION", "false").lower() == "true"
    )
    enable_delta_optimize_write: bool = field(
        default_factory=lambda: os.getenv("DELTA_OPTIMIZE_WRITE", "true").lower() == "true"
    )
    enable_delta_auto_compact: bool = field(
        default_factory=lambda: os.getenv("DELTA_AUTO_COMPACT", "true").lower() == "true"
    )


@dataclass(frozen=True)
class FraudRuleThresholds:
    """Thresholds for deterministic rule evaluation and composite risk scoring."""

    # High amount spike parameters
    amount_spike_multiplier: float = field(
        default_factory=lambda: float(os.getenv("AMOUNT_SPIKE_MULTIPLIER", "5.0"))
    )
    minimum_spike_absolute_amount: float = field(
        default_factory=lambda: float(os.getenv("MIN_SPIKE_ABS_AMOUNT", "500.0"))
    )

    # Velocity burst parameters
    velocity_5m_txn_count_threshold: int = field(
        default_factory=lambda: int(os.getenv("VELOCITY_5M_TXN_THRESHOLD", "3"))
    )
    velocity_1h_txn_count_threshold: int = field(
        default_factory=lambda: int(os.getenv("VELOCITY_1H_TXN_THRESHOLD", "8"))
    )
    velocity_24h_amount_threshold: float = field(
        default_factory=lambda: float(os.getenv("VELOCITY_24H_AMOUNT_THRESHOLD", "10000.0"))
    )

    # Rapid card testing parameters (small amount burst)
    card_testing_max_amount: float = field(
        default_factory=lambda: float(os.getenv("CARD_TESTING_MAX_AMOUNT", "5.0"))
    )
    card_testing_time_window_sec: int = field(
        default_factory=lambda: int(os.getenv("CARD_TESTING_WINDOW_SEC", "120"))
    )

    # Nocturnal transaction parameters (hours 01:00 - 04:59 local time)
    nocturnal_hours: Set[int] = field(
        default_factory=lambda: {1, 2, 3, 4}
    )
    nocturnal_amount_threshold: float = field(
        default_factory=lambda: float(os.getenv("NOCTURNAL_AMOUNT_THRESHOLD", "1500.0"))
    )

    # Risk score quarantine cutoff (0 - 100)
    quarantine_risk_score_threshold: int = field(
        default_factory=lambda: int(os.getenv("QUARANTINE_RISK_THRESHOLD", "80"))
    )
    high_risk_score_threshold: int = field(
        default_factory=lambda: int(os.getenv("HIGH_RISK_THRESHOLD", "60"))
    )


@dataclass(frozen=True)
class GeoVelocityConfig:
    """Geo-spatial parameters for impossible travel detection."""

    max_human_travel_speed_kmh: float = field(
        default_factory=lambda: float(os.getenv("MAX_TRAVEL_SPEED_KMH", "800.0"))
    )
    min_distance_for_speed_check_km: float = field(
        default_factory=lambda: float(os.getenv("MIN_DISTANCE_KM", "50.0"))
    )
    earth_radius_km: float = 6371.0
    high_risk_country_codes: List[str] = field(
        default_factory=lambda: os.getenv(
            "HIGH_RISK_COUNTRIES", "RU,NG,KP,IR,SY"
        ).split(",")
    )


@dataclass(frozen=True)
class MLModelConfig:
    """Spark MLlib model training and evaluation hyperparameters."""

    model_output_path: str = field(
        default_factory=lambda: os.getenv(
            "ML_MODEL_PATH",
            str(Path(__file__).resolve().parent.parent / "data" / "models" / "gbt_fraud_model")
        )
    )
    train_test_split_ratio: float = 0.8
    random_seed: int = 42
    max_iter: int = 20
    max_depth: int = 5
    probability_threshold: float = 0.5


@dataclass(frozen=True)
class AppConfig:
    """Master application configuration aggregating all sub-configs."""

    delta: DeltaPathConfig = field(default_factory=DeltaPathConfig)
    spark: SparkTuningConfig = field(default_factory=SparkTuningConfig)
    rules: FraudRuleThresholds = field(default_factory=FraudRuleThresholds)
    geo: GeoVelocityConfig = field(default_factory=GeoVelocityConfig)
    ml: MLModelConfig = field(default_factory=MLModelConfig)


# Singleton instance for global import
config = AppConfig()
