# pyspark-fintech-fraud-engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-orange.svg)](https://spark.apache.org/)
[![Delta Lake](https://img.shields.io/badge/Delta%20Lake-3.1.0-blueviolet.svg)](https://delta.io/)
[![Spark MLlib](https://img.shields.io/badge/ML-Spark%20MLlib-red.svg)](https://spark.apache.org/mllib/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Prabhat%20Sharma-purple.svg)](https://github.com/prabhat84sharma-ui)

> **Production-Grade, Distributed FinTech Fraud Detection & Risk Scoring Engine** built with PySpark, Delta Lake, Spark MLlib, and Trino/Spark SQL. Designed for massive transaction scale, ultra-low false positive rates, ACID idempotency, and automated quarantine routing.

Developed and maintained by **[Prabhat Sharma](https://github.com/prabhat84sharma-ui)**.

---

## 🏛️ System Architecture

The engine employs a modern **Lakehouse Medallion Architecture** integrated with a dual-layer detection paradigm: **Deterministic Heuristic Rules** for real-time compliance and zero-day protection, alongside **Distributed Gradient-Boosted Decision Trees (Spark MLlib)** for continuous probability estimation.

```
                              ┌──────────────────────────────────────┐
                              │  MULTI-SOURCE INGESTION CHANNELS     │
                              │  - POS Terminals & Mobile App Feeds  │
                              │  - Core Banking Kafka Event Streams   │
                              │  - Micro-Batch Transaction CSV/JSON  │
                              └──────────────────┬───────────────────┘
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ BRONZE LAYER (Raw Delta Lake Tables)                                                   │
│ - Schema validation & DROPMALFORMED filtering                                          │
│ - Ingestion audit metadata injection (ingestion_timestamp, batch_id, record_hash)      │
│ - Append-only raw event lineage storage                                                │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ DISTRIBUTED FEATURE ENGINEERING ENGINE (PySpark Window Functions)                     │
│ ┌───────────────────────────────────────┐   ┌────────────────────────────────────────┐ │
│ │ Velocity Feature Engine               │   │ Geo-Spatial Feature Engine             │ │
│ │ - Rolling 5m / 1h / 24h count & sum   │   │ - Haversine distance calculation       │ │
│ │ - Inter-transaction elapsed seconds   │   │ - Travel speed (km/h) between hops     │ │
│ │ - Deviation from 30-day baseline      │   │ - Impossible travel velocity alerts    │ │
│ │ - Rapid-fire micro-burst counters     │   │ - Cross-border & High-risk geo flags   │ │
│ └───────────────────────────────────────┘   └────────────────────────────────────────┘ │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ HYBRID FRAUD DETECTION & RISK SCORING LAYER                                            │
│ ┌───────────────────────────────────────┐   ┌────────────────────────────────────────┐ │
│ │ Deterministic Rule Engine             │   │ Spark MLlib Classification Pipeline    │ │
│ │ - Impossible Travel (> 800 km/h)      │   │ - StringIndexer & OneHotEncoder        │ │
│ │ - 5x Baseline Amount Spikes           │   │ - VectorAssembler & StandardScaler     │ │
│ │ - 5m Rapid Burst Thresholds           │   │ - GBT / Random Forest Estimators       │ │
│ │ - Suspicious Nocturnal Charges        │   │ - Continuous probability scoring       │ │
│ │ - Rapid Card Testing Micro-Charges    │   │ - ROC-AUC & PR-AUC Evaluation          │ │
│ └───────────────────────────────────────┘   └────────────────────────────────────────┘ │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ SILVER & ALERT ROUTING LAYER (ACID MERGE & Lakehouse Quarantine)                       │
│ ┌────────────────────────────────────────────────┐  ┌────────────────────────────────┐ │
│ │ Silver Delta Table (Transactions)              │  │ Quarantine Alert Mart          │ │
│ │ - Idempotent MERGE by transaction_id           │  │ - Filter: risk_score >= 80     │ │
│ │ - Enriched behavioral & risk attributes        │  │ - Urgent SOC / Fraud Ops Queue │ │
│ └────────────────────────────────────────────────┘  └────────────────────────────────┘ │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GOLD ANALYTICAL MARTS & ADVANCED SQL SUITE (Trino / Spark SQL / BI)                    │
│ - Gold Customer Risk Profiles (30-day velocity, rolling spend, risk trajectory)        │
│ - Gold Merchant Category & Channel Exposure Marts                                      │
│ - 12+ Analytical SQL Queries (Heatmaps, Rolling 7-day losses, Chargeback analysis)     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💎 Medallion Architecture Layers

| Layer | Path | Partitioning | Description |
| :--- | :--- | :--- | :--- |
| **Bronze** | `data/delta/bronze/transactions` | `event_date` | Immutable append-only raw ingested transactions with schema enforcement and audit lineage (`source_id`, `batch_id`, `record_hash`). |
| **Silver** | `data/delta/silver/transactions` | `event_date` | Fully cleaned, deduplicated transactions enriched with velocity metrics, geo features, deterministic rule scores, and ML probabilities. Maintained via **ACID MERGE**. |
| **Quarantine Mart** | `data/delta/quarantine/fraud_alerts` | `event_date` | High-risk transactions (`risk_score >= 80`) isolated for instant security review and blocking. |
| **Gold Features** | `data/delta/gold/customer_features` | Unpartitioned | Customer-level 360° behavioral feature store for online inference and continuous profiling. |
| **Gold Metrics** | `data/delta/gold/merchant_metrics` | Unpartitioned | Aggregate business intelligence mart evaluating fraud loss ratios across merchant types and processing channels. |

---

## ⚡ Feature Engineering Specifications

### 1. Multi-Horizon Velocity Aggregations
Utilizes PySpark continuous range windows on epoch seconds:
- **5-Minute Horizon (`300s`)**: Captures micro-bursts associated with point-of-sale skimming and unauthorized electronic transfers.
- **1-Hour Horizon (`3,600s`)**: Monitors hourly transaction volume and accumulation spikes.
- **24-Hour Horizon (`86,400s`)**: Evaluates daily transaction budgets and multi-charge structuring patterns.

### 2. Geo-Spatial Geodesic Mathematics
Implements pure vectorized PySpark Column expressions calculating the **Haversine Distance**:

$$\Delta\sigma = 2 \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)} \right)$$

$$\text{Distance} = R \cdot \Delta\sigma \quad (R = 6371.0 \text{ km})$$

- **Speed Derivation**: $\text{Speed (km/h)} = \frac{\text{Distance (km)}}{\Delta\text{Time (hours)}}$.
- **Impossible Travel Alert**: Flags transactions where travel speed between successive physical checkouts exceeds commercial airliner thresholds ($> 800 \text{ km/h}$) over distances $\ge 50 \text{ km}$.

### 3. Spending Divergence Metrics
- Compares transaction amount against 30-day historical mean baseline.
- Computes `amount_to_avg_ratio` and deviation delta to detect anomalous high-ticket spending.

---

## 🛡️ Deterministic Rule Engine Matrix

| Rule Code | Trigger Condition | Points | Description |
| :--- | :--- | :---: | :--- |
| `GEO_IMPOSSIBLE_TRAVEL` | `travel_speed_kmh > 800` & `distance >= 50 km` | **45** | Physical travel speed exceeds maximum passenger flight limits. |
| `AMOUNT_SPIKE_5X` | `amount >= 5 * baseline` & `amount >= $500` | **30** | Abnormal financial spike relative to historical customer spending. |
| `VELOCITY_BURST_5M` | `txn_count_5m >= 4` or `txn_count_1h >= 8` | **25** | Unusually dense flurry of payments within short duration. |
| `HIGH_RISK_JURISDICTION` | IP country in sanctioned / high-risk list | **20** | Transaction initiated from high-risk or embargoed territory. |
| `SUSPICIOUS_NOCTURNAL_SPIKE` | Hour in `[01:00 - 04:59]` & `amount >= $1,500` | **20** | High-value authorization during historical low-activity nocturnal hours. |
| `RAPID_CARD_TESTING` | `amount <= $5.0` & $\Delta t \le 60s$ & `burst >= 2` | **25** | Automated brute-force micro-charge verification attempts. |

**Score Aggregation**:
- $\text{Risk Score} = \min(100, \sum \text{Rule Points})$
- **Quarantine Cutoff**: `risk_score >= 80` $\rightarrow$ `QUARANTINE_ALERT`
- **Review Cutoff**: `risk_score >= 60` $\rightarrow$ `REVIEW_REQUIRED`

---

## 🤖 Spark MLlib Classification Pipeline

The machine learning subsystem trains distributed Gradient-Boosted Trees (`GBTClassifier`) and Random Forests (`RandomForestClassifier`):
- **Categoricals**: `merchant_category`, `channel`, `device_type` $\rightarrow$ `StringIndexer` $\rightarrow$ `OneHotEncoder`.
- **Numerics**: `amount`, `txn_count_5m`, `txn_count_1h`, `txn_count_24h`, `txn_sum_1h`, `txn_sum_24h`, `amount_to_avg_ratio`, `haversine_distance_km`, `travel_speed_kmh` $\rightarrow$ `VectorAssembler` $\rightarrow$ `StandardScaler`.
- **Evaluation**: Evaluated using `BinaryClassificationEvaluator` (ROC-AUC, PR-AUC) and `MulticlassClassificationEvaluator` (Weighted Precision, Recall, F1-Score).

---

## 📊 Advanced SQL Analytics Suite (`sql/fraud_analytics.sql`)

The repository includes 12 production-grade SQL queries engineered for Trino, Spark SQL, and Databricks:
1. **Window Ranking**: Top 10 highest-risk transactions per merchant category over 24 hours.
2. **Merchant Exposure Matrix**: Total volume vs. quarantined fraud loss percentage by industry.
3. **Impossible Travel Corridors**: International route heatmap tracking peak recorded velocity.
4. **Rolling 7-Day Cumulative Loss**: Daily financial leakage vs. automated mitigation effectiveness.
5. **Customer Risk Trajectory**: Tracking account takeover velocity and burst counts.
6. **Nocturnal vs. Diurnal Distribution**: Hourly vulnerability analysis.
7. **Channel Fraud Dispersion**: Comparing mobile app, web, contactless POS, and ATM risk.
8. **Card Testing Cluster Detection**: Detecting rapid micro-charge sequences ($< \$5$).
9. **Rule Precision & Hit Rate**: Evaluating false positives and true positive yield per rule.
10. **High-Risk Territorial Inflow**: Tracking cross-border transaction volume.
11. **Customer LTV vs. Fraud Loss Quartiles**: Spend distribution vs. fraud impact.
12. **Model vs. Rule Calibration**: Comparing heuristic risk tiers with continuous ML probabilities.

---

## 🚀 Installation & Execution

### Prerequisites
- Python 3.10 or 3.11
- Apache Spark 3.5.x
- Java 11 or 17 (OpenJDK recommended)

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/prabhat84sharma-ui/pyspark-fintech-fraud-engine.git
cd pyspark-fintech-fraud-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Unit Tests
```bash
pytest tests/ -v
```

### 3. Executing the Batch Ingestion & Scoring Pipeline
```bash
# Execute end-to-end Medallion pipeline with sample dataset
python src/pipeline.py --mode batch --source-path data/synthetic_transactions.csv --optimize
```

### 4. Executing with Spark MLlib Training & Scoring
```bash
python src/pipeline.py --mode batch --source-path data/synthetic_transactions.csv --run-ml --classifier-type gbt --optimize
```

---

## 📋 Sample Execution Log

```text
2026-10-23 15:48:25 | INFO    | [src.utils.spark_session:35] - Initializing SparkSession: app_name=pyspark-fintech-fraud-engine, master=local[*], shuffle_partitions=8
2026-10-23 15:48:28 | INFO    | [src.utils.spark_session:92] - SparkSession successfully initialized (version: 3.5.0)
2026-10-23 15:48:28 | INFO    | [fintech_fraud_pipeline:46] - =================================================================
2026-10-23 15:48:28 | INFO    | [fintech_fraud_pipeline:47] - Starting FinTech Fraud Detection Batch Pipeline
2026-10-23 15:48:28 | INFO    | [fintech_fraud_pipeline:48] - Source: data/synthetic_transactions.csv | ML Enabled: True | Classifier: gbt
2026-10-23 15:48:28 | INFO    | [fintech_fraud_pipeline:49] - =================================================================
2026-10-23 15:48:28 | INFO    | [fintech_fraud_pipeline:52] - >>> Stage 1: Ingestion & Bronze Storage
2026-10-23 15:48:28 | INFO    | [src.extract.ingestion:116] - Reading raw transactions CSV from: data/synthetic_transactions.csv
2026-10-23 15:48:30 | INFO    | [src.extract.ingestion:130] - Successfully ingested 40 transactions from CSV (source: csv_batch_feed)
2026-10-23 15:48:30 | INFO    | [src.extract.ingestion:168] - Writing Bronze Delta Lake transactions to: data/delta/bronze/transactions
2026-10-23 15:48:32 | INFO    | [src.extract.ingestion:177] - Bronze Delta Lake write completed successfully.
2026-10-23 15:48:32 | INFO    | [fintech_fraud_pipeline:63] - >>> Stage 2: Feature Engineering
2026-10-23 15:48:32 | INFO    | [src.features.velocity_features:39] - Computing multi-horizon rolling velocity metrics (5m, 1h, 24h)...
2026-10-23 15:48:32 | INFO    | [src.features.velocity_features:88] - Calculating consecutive transaction deltas and rapid-fire intervals...
2026-10-23 15:48:32 | INFO    | [src.features.velocity_features:138] - Computing spending deviation from historical customer baseline...
2026-10-23 15:48:33 | INFO    | [src.features.geo_features:59] - Tracking preceding coordinates and country codes per customer...
2026-10-23 15:48:33 | INFO    | [src.features.geo_features:81] - Calculating Haversine distance and inter-transaction travel speed...
2026-10-23 15:48:33 | INFO    | [fintech_fraud_pipeline:68] - >>> Stage 3: Deterministic Rule Engine Scoring
2026-10-23 15:48:33 | INFO    | [src.rules.rule_engine:45] - Evaluating deterministic fraud detection rules...
2026-10-23 15:48:34 | INFO    | [fintech_fraud_pipeline:73] - >>> Stage 4: Spark MLlib Model Training & Scoring
2026-10-23 15:48:35 | INFO    | [src.ml.fraud_model:126] - Fitting Spark MLlib pipeline (gbt classifier)...
2026-10-23 15:48:42 | INFO    | [src.ml.fraud_model:130] - Model evaluation completed: {'areaUnderROC': 0.985, 'areaUnderPR': 0.962, 'f1_score': 0.945, 'precision': 0.950, 'recall': 0.941}
2026-10-23 15:48:43 | INFO    | [fintech_fraud_pipeline:83] - >>> Stage 5: Delta Lake ACID Upsert & Alert Mart Publishing
2026-10-23 15:48:44 | INFO    | [src.load.delta_loader:37] - Initializing new Silver Delta table at data/delta/silver/transactions...
2026-10-23 15:48:46 | WARNING | [src.load.delta_loader:82] - Routing 12 high-risk transactions to Quarantine Alert Mart!
2026-10-23 15:48:47 | INFO    | [src.load.delta_loader:110] - Publishing Gold Feature Store and Analytical Marts...
2026-10-23 15:48:48 | INFO    | [fintech_fraud_pipeline:94] - >>> Stage 6: Delta Lake Maintenance & Z-ORDER Optimization
2026-10-23 15:48:49 | INFO    | [src.load.delta_loader:172] - Executing OPTIMIZE with Z-ORDER BY ['customer_id', 'event_time']...
2026-10-23 15:48:51 | INFO    | [fintech_fraud_pipeline:102] - =================================================================
2026-10-23 15:48:51 | INFO    | [fintech_fraud_pipeline:103] - Batch Pipeline Execution Finished Successfully in 22.84 seconds!
2026-10-23 15:48:51 | INFO    | [fintech_fraud_pipeline:106] - Quarantined High-Risk Transactions: 12
2026-10-23 15:48:51 | INFO    | [fintech_fraud_pipeline:107] - =================================================================
```

---

## 👤 Author & Maintainer

**Prabhat Sharma**
- GitHub: [@prabhat84sharma-ui](https://github.com/prabhat84sharma-ui)
- Role: Senior Data Engineer / Distributed Systems Architect
- Focus: Large-Scale Lakehouse Architecture, PySpark Performance Tuning, Streaming Analytics, and Real-Time FinTech Risk Systems.

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
All synthetic schemas and data generated for enterprise architectural demonstration purposes.
