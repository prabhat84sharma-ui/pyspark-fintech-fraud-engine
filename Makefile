.PHONY: help install test lint format clean run-pipeline run-ml optimize-delta

PYTHON ?= python
PYTEST ?= pytest

help:
	@echo "=========================================================================="
	@echo "pyspark-fintech-fraud-engine - Large-Scale Fraud Detection Engine"
	@echo "Author: Prabhat Sharma (https://github.com/prabhat84sharma-ui)"
	@echo "=========================================================================="
	@echo "Available commands:"
	@echo "  make install        Install project dependencies"
	@echo "  make test           Run unit tests using pytest"
	@echo "  make lint           Check code quality using flake8 and mypy"
	@echo "  make format         Auto-format code using black"
	@echo "  make clean          Clean temporary files and Delta artifacts"
	@echo "  make run-pipeline   Execute full end-to-end ingestion and scoring pipeline"
	@echo "  make run-ml         Train and evaluate Spark MLlib fraud detection model"
	@echo "  make optimize-delta Run Delta Lake compaction and Z-ORDER optimization"
	@echo "=========================================================================="

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTEST) tests/ -v

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503,E402
	mypy src/

format:
	black src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache metastore_db derby.log spark-warehouse data/delta
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-pipeline:
	$(PYTHON) src/pipeline.py --mode batch --source-path data/synthetic_transactions.csv --optimize

run-ml:
	$(PYTHON) src/pipeline.py --mode batch --source-path data/synthetic_transactions.csv --run-ml

optimize-delta:
	$(PYTHON) src/pipeline.py --optimize-only
