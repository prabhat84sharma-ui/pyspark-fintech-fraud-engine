"""Multi-source data ingestion engine for FinTech banking and payment transactions.

Supports batch and structured streaming ingestion from CSV, JSON, and Parquet,
enforcing strict schemas, injecting audit metadata, and managing Bronze Delta Lake writes.
"""

from __future__ import annotations

import uuid
from typing import Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_raw_transaction_schema() -> StructType:
    """Returns the explicit canonical schema for incoming raw FinTech transactions."""
    return StructType(
        [
            StructField("transaction_id", StringType(), nullable=False),
            StructField("customer_id", StringType(), nullable=False),
            StructField("card_id", StringType(), nullable=True),
            StructField("account_id", StringType(), nullable=True),
            StructField("event_time", StringType(), nullable=False),
            StructField("amount", DoubleType(), nullable=False),
            StructField("currency", StringType(), nullable=True),
            StructField("merchant_id", StringType(), nullable=True),
            StructField("merchant_name", StringType(), nullable=True),
            StructField("merchant_category", StringType(), nullable=True),
            StructField("channel", StringType(), nullable=True),
            StructField("device_type", StringType(), nullable=True),
            StructField("device_id", StringType(), nullable=True),
            StructField("ip_address", StringType(), nullable=True),
            StructField("ip_country", StringType(), nullable=True),
            StructField("latitude", DoubleType(), nullable=True),
            StructField("longitude", DoubleType(), nullable=True),
            StructField("historical_avg_amount_30d", DoubleType(), nullable=True),
            StructField("is_fraud_label", IntegerType(), nullable=True),
        ]
    )


class TransactionIngestionEngine:
    """Ingestion pipeline handler responsible for reading, validating, and enriching raw feeds."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.schema = get_raw_transaction_schema()

    def inject_audit_metadata(
        self, df: DataFrame, source_id: str, batch_id: Optional[str] = None
    ) -> DataFrame:
        """Injects lineage and audit metadata columns into the ingested DataFrame.

        Added columns:
        - ingestion_timestamp: Current UTC timestamp of ingestion.
        - source_id: Identifier of the source feed / queue / file.
        - batch_id: Unique GUID tracking the ingestion run.
        - record_hash: MD5 hash of customer_id, event_time, and amount for duplicate auditing.
        """
        resolved_batch_id = batch_id or str(uuid.uuid4())

        return (
            df.withColumn("ingestion_timestamp", F.current_timestamp())
            .withColumn("source_id", F.lit(source_id))
            .withColumn("batch_id", F.lit(resolved_batch_id))
            .withColumn(
                "record_hash",
                F.md5(
                    F.concat_ws(
                        "||",
                        F.col("transaction_id"),
                        F.col("customer_id"),
                        F.col("event_time"),
                        F.col("amount").cast("string"),
                    )
                ),
            )
        )

    def standardize_event_timestamp(self, df: DataFrame) -> DataFrame:
        """Parses event_time string to UTC TimestampType and extracts partition date."""
        return (
            df.withColumn(
                "event_time",
                F.to_timestamp(F.col("event_time"), "yyyy-MM-dd HH:mm:ss"),
            )
            .withColumn("event_date", F.to_date(F.col("event_time")))
            .filter(F.col("event_time").isNotNull() & F.col("transaction_id").isNotNull())
        )

    def read_csv(
        self,
        file_path: str,
        source_id: str = "csv_batch_feed",
        batch_id: Optional[str] = None,
    ) -> DataFrame:
        """Reads CSV transaction feed with schema enforcement and audit metadata injection.

        Args:
            file_path: Path or glob pattern for the CSV file(s).
            source_id: Logical source name for audit tracking.
            batch_id: Optional batch execution identifier.

        Returns:
            DataFrame adhering to canonical Bronze schema.
        """
        logger.info("Reading raw transactions CSV from: %s", file_path)
        raw_df = (
            self.spark.read.format("csv")
            .option("header", "true")
            .option("mode", "DROPMALFORMED")
            .schema(self.schema)
            .load(file_path)
        )

        standardized_df = self.standardize_event_timestamp(raw_df)
        enriched_df = self.inject_audit_metadata(standardized_df, source_id, batch_id)

        count = enriched_df.count()
        logger.info("Successfully ingested %d transactions from CSV (source: %s)", count, source_id)
        return enriched_df

    def read_json_stream(
        self,
        directory_path: str,
        source_id: str = "json_event_stream",
    ) -> DataFrame:
        """Reads streaming JSON micro-batches from a drop folder or message queue spill."""
        logger.info("Configuring streaming JSON reader from directory: %s", directory_path)
        streaming_df = (
            self.spark.readStream.format("json")
            .schema(self.schema)
            .option("maxFilesPerTrigger", 5)
            .load(directory_path)
        )

        standardized_df = self.standardize_event_timestamp(streaming_df)
        return self.inject_audit_metadata(standardized_df, source_id=source_id)

    def read_parquet(
        self,
        file_path: str,
        source_id: str = "parquet_lake_feed",
        batch_id: Optional[str] = None,
    ) -> DataFrame:
        """Reads Parquet transaction feed."""
        logger.info("Reading raw transactions Parquet from: %s", file_path)
        raw_df = self.spark.read.format("parquet").schema(self.schema).load(file_path)
        standardized_df = self.standardize_event_timestamp(raw_df)
        return self.inject_audit_metadata(standardized_df, source_id, batch_id)

    def write_to_bronze(
        self,
        df: DataFrame,
        bronze_path: Optional[str] = None,
        partition_by: str = "event_date",
    ) -> None:
        """Writes ingested transactions to Bronze Delta Lake layer in append mode."""
        destination = bronze_path or config.delta.bronze_path
        logger.info("Writing Bronze Delta Lake transactions to: %s", destination)

        (
            df.write.format("delta")
            .mode("append")
            .partitionBy(partition_by)
            .option("mergeSchema", "true")
            .save(destination)
        )
        logger.info("Bronze Delta Lake write completed successfully.")
