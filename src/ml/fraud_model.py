"""Spark MLlib machine learning pipeline for FinTech fraud prediction and risk scoring.

Implements end-to-end distributed ML workflows:
- Categorical indexing and one-hot encoding (merchant_category, channel, device_type)
- Numeric scaling (amount, multi-horizon velocity, Haversine distance, speed)
- GBTClassifier / RandomForestClassifier training
- Comprehensive evaluation (ROC-AUC, PR-AUC, Precision, Recall, F1)
- Model persistence and batch/streaming inference scoring
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import config, MLModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudMLPipeline:
    """End-to-end Spark MLlib Pipeline for training and scoring fraud detection models."""

    CATEGORICAL_COLS = ["merchant_category", "channel", "device_type"]
    NUMERIC_COLS = [
        "amount",
        "txn_count_5m",
        "txn_count_1h",
        "txn_count_24h",
        "txn_sum_1h",
        "txn_sum_24h",
        "amount_to_avg_ratio",
        "haversine_distance_km",
        "travel_speed_kmh",
    ]

    def __init__(self, ml_config: Optional[MLModelConfig] = None):
        self.config = ml_config or config.ml
        self.model: Optional[PipelineModel] = None

    def prepare_data(self, df: DataFrame, label_col: str = "is_fraud_label") -> DataFrame:
        """Cleans and imputes missing values for feature assembly."""
        cat_defaults = {col: "unknown" for col in self.CATEGORICAL_COLS}
        num_defaults = {col: 0.0 for col in self.NUMERIC_COLS}

        cleaned = df.na.fill(cat_defaults).na.fill(num_defaults)

        # Cast numerics explicitly to double
        for col_name in self.NUMERIC_COLS:
            if col_name in cleaned.columns:
                cleaned = cleaned.withColumn(col_name, F.col(col_name).cast("double"))

        # Cast label to double if present
        if label_col in cleaned.columns:
            cleaned = cleaned.withColumn(label_col, F.col(label_col).cast("double"))

        return cleaned

    def build_pipeline(
        self, classifier_type: str = "gbt", label_col: str = "is_fraud_label"
    ) -> Pipeline:
        """Constructs Spark ML pipeline stages."""
        stages = []

        # Categorical processing: StringIndexer -> OneHotEncoder
        indexed_cols = [f"{col}_idx" for col in self.CATEGORICAL_COLS]
        encoded_cols = [f"{col}_vec" for col in self.CATEGORICAL_COLS]

        indexer = StringIndexer(
            inputCols=self.CATEGORICAL_COLS,
            outputCols=indexed_cols,
            handleInvalid="keep",
        )
        encoder = OneHotEncoder(
            inputCols=indexed_cols,
            outputCols=encoded_cols,
            handleInvalid="keep",
        )
        stages.extend([indexer, encoder])

        # Assemble numeric features and scale them
        num_assembler = VectorAssembler(
            inputCols=self.NUMERIC_COLS,
            outputCol="numeric_features_raw",
            handleInvalid="keep",
        )
        scaler = StandardScaler(
            inputCol="numeric_features_raw",
            outputCol="numeric_features_scaled",
            withStd=True,
            withMean=False,
        )
        stages.extend([num_assembler, scaler])

        # Combine all features into final vector
        final_assembler = VectorAssembler(
            inputCols=encoded_cols + ["numeric_features_scaled"],
            outputCol="features",
            handleInvalid="keep",
        )
        stages.append(final_assembler)

        # Choose classifier
        if classifier_type.lower() == "rf":
            classifier = RandomForestClassifier(
                labelCol=label_col,
                featuresCol="features",
                numTrees=30,
                maxDepth=self.config.max_depth,
                seed=self.config.random_seed,
            )
        else:
            classifier = GBTClassifier(
                labelCol=label_col,
                featuresCol="features",
                maxIter=self.config.max_iter,
                maxDepth=self.config.max_depth,
                seed=self.config.random_seed,
            )

        stages.append(classifier)
        return Pipeline(stages=stages)

    def train(
        self,
        df: DataFrame,
        label_col: str = "is_fraud_label",
        classifier_type: str = "gbt",
    ) -> Tuple[PipelineModel, Dict[str, float]]:
        """Fits the pipeline on input data and returns model and evaluation metrics."""
        logger.info("Preparing data for ML training and evaluation...")
        cleaned = self.prepare_data(df, label_col)

        train_ratio = self.config.train_test_split_ratio
        test_ratio = 1.0 - train_ratio
        train_df, test_df = cleaned.randomSplit(
            [train_ratio, test_ratio], seed=self.config.random_seed
        )

        train_count = train_df.count()
        test_count = test_df.count()
        logger.info("Dataset split: Train=%d, Test=%d", train_count, test_count)

        pipeline = self.build_pipeline(classifier_type=classifier_type, label_col=label_col)
        logger.info("Fitting Spark MLlib pipeline (%s classifier)...", classifier_type)
        self.model = pipeline.fit(train_df)

        metrics = self.evaluate(test_df, label_col=label_col)
        logger.info("Model evaluation completed: %s", metrics)
        return self.model, metrics

    def evaluate(
        self, test_df: DataFrame, label_col: str = "is_fraud_label"
    ) -> Dict[str, float]:
        """Evaluates trained model on a test DataFrame across multiple metrics."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() or load_model() first.")

        prepared_test = self.prepare_data(test_df, label_col)
        predictions = self.model.transform(prepared_test)

        # Binary evaluators
        roc_evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol=label_col,
            metricName="areaUnderROC",
        )
        pr_evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol=label_col,
            metricName="areaUnderPR",
        )

        # Multiclass evaluators for Precision, Recall, F1
        f1_evaluator = MulticlassClassificationEvaluator(
            predictionCol="prediction",
            labelCol=label_col,
            metricName="f1",
        )
        prec_evaluator = MulticlassClassificationEvaluator(
            predictionCol="prediction",
            labelCol=label_col,
            metricName="weightedPrecision",
        )
        recall_evaluator = MulticlassClassificationEvaluator(
            predictionCol="prediction",
            labelCol=label_col,
            metricName="weightedRecall",
        )

        roc_auc = roc_evaluator.evaluate(predictions)
        pr_auc = pr_evaluator.evaluate(predictions)
        f1_score = f1_evaluator.evaluate(predictions)
        precision = prec_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)

        return {
            "areaUnderROC": round(roc_auc, 4),
            "areaUnderPR": round(pr_auc, 4),
            "f1_score": round(f1_score, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    def predict(self, df: DataFrame) -> DataFrame:
        """Applies trained model to generate fraud probability and predicted class."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() or load_model() first.")

        prepared = self.prepare_data(df)
        predictions = self.model.transform(prepared)

        # Extract probability of fraud (class 1)
        # GBTClassifier/RF outputs a vector of probabilities [P(0), P(1)]
        scored = (
            predictions
            .withColumn("prob_array", vector_to_array(F.col("probability")))
            .withColumn("ml_fraud_probability", F.round(F.col("prob_array")[1], 4))
            .withColumn("ml_prediction", F.col("prediction").cast("integer"))
            .drop("prob_array", "numeric_features_raw", "numeric_features_scaled", "features", "rawPrediction", "probability")
        )

        return scored

    def save_model(self, path: Optional[str] = None) -> str:
        """Persists trained PipelineModel to disk or cloud object storage."""
        if self.model is None:
            raise ValueError("No model available to save.")
        target_path = path or self.config.model_output_path
        logger.info("Saving trained model to: %s", target_path)
        self.model.write().overwrite().save(target_path)
        return target_path

    def load_model(self, path: Optional[str] = None) -> PipelineModel:
        """Loads a persisted PipelineModel."""
        target_path = path or self.config.model_output_path
        logger.info("Loading model from: %s", target_path)
        self.model = PipelineModel.load(target_path)
        return self.model
