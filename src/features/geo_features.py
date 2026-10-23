"""Geo-spatial feature engineering and impossible travel detection for PySpark.

Implements pure PySpark SQL / math functions for Haversine geodesic distance calculation,
velocity speed tracking between successive locations, and geo-risk classification.
"""

from __future__ import annotations

from typing import List, Optional
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.config import config, GeoVelocityConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeoFeatureEngine:
    """Computes geographical distance, inter-city velocity, and high-risk geo flags."""

    def __init__(self, geo_config: Optional[GeoVelocityConfig] = None):
        self.geo_config = geo_config or config.geo

    @staticmethod
    def calculate_haversine_distance(
        lat1_col: str, lon1_col: str, lat2_col: str, lon2_col: str, earth_radius_km: float = 6371.0
    ) -> F.Column:
        """Pure PySpark SQL Column expression calculating Haversine distance in kilometers.

        dlat = rad(lat2 - lat1)
        dlon = rad(lon2 - lon1)
        a = sin^2(dlat/2) + cos(rad(lat1)) * cos(rad(lat2)) * sin^2(dlon/2)
        c = 2 * asin(sqrt(clip(a, 0, 1)))
        distance = R * c
        """
        dlat = F.radians(F.col(lat2_col) - F.col(lat1_col))
        dlon = F.radians(F.col(lon2_col) - F.col(lon1_col))
        lat1_rad = F.radians(F.col(lat1_col))
        lat2_rad = F.radians(F.col(lat2_col))

        # a calculation
        a_term = (
            (F.sin(dlat / 2.0) ** 2)
            + (F.cos(lat1_rad) * F.cos(lat2_rad) * (F.sin(dlon / 2.0) ** 2))
        )

        # Clip bounds to avoid NaN due to floating point precision
        clipped_a = F.least(F.lit(1.0), F.greatest(F.lit(0.0), a_term))
        c_term = 2.0 * F.asin(F.sqrt(clipped_a))

        return F.round(F.lit(earth_radius_km) * c_term, 2)

    def compute_consecutive_geo_deltas(self, df: DataFrame) -> DataFrame:
        """Tracks coordinates, timestamps, and country codes of preceding transaction."""
        logger.info("Tracking preceding coordinates and country codes per customer...")

        if "event_time_epoch" not in df.columns:
            df = df.withColumn("event_time_epoch", F.col("event_time").cast("long"))

        w_cust_order = Window.partitionBy("customer_id").orderBy("event_time")

        return (
            df
            .withColumn("prev_latitude", F.lag("latitude", 1).over(w_cust_order))
            .withColumn("prev_longitude", F.lag("longitude", 1).over(w_cust_order))
            .withColumn("prev_ip_country", F.lag("ip_country", 1).over(w_cust_order))
            .withColumn(
                "prev_event_time_epoch",
                F.coalesce(
                    F.lag("event_time_epoch", 1).over(w_cust_order),
                    F.col("event_time_epoch"),
                ),
            )
        )

    def compute_travel_speed_and_anomalies(self, df: DataFrame) -> DataFrame:
        """Calculates distance, elapsed hours, physical speed, and impossible travel flags."""
        logger.info("Calculating Haversine distance and inter-transaction travel speed...")

        # Calculate Haversine distance when preceding coords are available
        has_both_coords = (
            F.col("latitude").isNotNull()
            & F.col("longitude").isNotNull()
            & F.col("prev_latitude").isNotNull()
            & F.col("prev_longitude").isNotNull()
        )

        dist_col = F.when(
            has_both_coords,
            self.calculate_haversine_distance(
                "prev_latitude", "prev_longitude", "latitude", "longitude"
            ),
        ).otherwise(F.lit(0.0))

        # Time elapsed in hours
        time_diff_sec_col = F.greatest(
            F.lit(1),
            F.col("event_time_epoch") - F.col("prev_event_time_epoch"),
        )
        time_diff_hours_col = time_diff_sec_col / 3600.0

        with_dist = df.withColumn("haversine_distance_km", dist_col).withColumn(
            "travel_elapsed_hours", F.round(time_diff_hours_col, 4)
        )

        # Travel speed (km/h)
        speed_col = F.when(
            (F.col("haversine_distance_km") > 0.0) & (F.col("travel_elapsed_hours") > 0.0),
            F.round(F.col("haversine_distance_km") / F.col("travel_elapsed_hours"), 2),
        ).otherwise(F.lit(0.0))

        with_speed = with_dist.withColumn("travel_speed_kmh", speed_col)

        # Impossible travel flag (exceeds supersonic airliner / passenger cruising speed)
        is_impossible = (
            (F.col("travel_speed_kmh") > self.geo_config.max_human_travel_speed_kmh)
            & (F.col("haversine_distance_km") >= self.geo_config.min_distance_for_speed_check_km)
        )

        # Cross-border jump flag
        is_cross_border = (
            F.col("prev_ip_country").isNotNull()
            & (F.col("ip_country") != F.col("prev_ip_country"))
        )

        # High-risk jurisdiction flag
        is_high_risk = F.col("ip_country").isin(self.geo_config.high_risk_country_codes)

        return (
            with_speed
            .withColumn("is_impossible_travel", F.when(is_impossible, F.lit(1)).otherwise(F.lit(0)))
            .withColumn("is_cross_border_txn", F.when(is_cross_border, F.lit(1)).otherwise(F.lit(0)))
            .withColumn("is_high_risk_country", F.when(is_high_risk, F.lit(1)).otherwise(F.lit(0)))
        )

    def transform(self, df: DataFrame) -> DataFrame:
        """Executes full geo feature transformation pipeline."""
        logger.info("Executing Geo-Spatial Feature transformation pipeline...")
        step1 = self.compute_consecutive_geo_deltas(df)
        step2 = self.compute_travel_speed_and_anomalies(step1)
        logger.info("Geo-Spatial Feature transformation complete.")
        return step2
