# Makefile for Customer Churn Prediction Project
# Windows-compatible commands

.PHONY: help setup install test train evaluate explain clean all

help:
	@echo "Customer Churn Prediction - Available Commands:"
	@echo ""
	@echo "  make setup      - Create directories and install dependencies"
	@echo "  make install    - Install Python dependencies"
	@echo "  make test       - Run unit tests"
	@echo "  make train      - Train models and generate comparison"
	@echo "  make evaluate   - Evaluate best model on test set"
	@echo "  make explain    - Generate SHAP analysis and business optimization"
	@echo "  make all        - Run complete pipeline (preprocess, train, evaluate, explain)"
	@echo "  make clean      - Remove generated files"
	@echo ""

setup:
	@echo Creating project directories...
	@if not exist data\processed mkdir data\processed
	@if not exist models mkdir models
	@if not exist reports\figures mkdir reports\figures
	@echo Done!

install:
	@echo Installing dependencies...
	python -m pip install -q -U pip
	python -m pip install -q pandas numpy scikit-learn matplotlib seaborn joblib "shap<0.50"
	@echo Dependencies installed!

test:
	@echo Running unit tests...
	python -m pytest tests/ -v || python -m unittest discover tests -v
	@echo Tests complete!

preprocess:
	@echo Running preprocessing pipeline...
	python -m src.churn.preprocess
	@echo Preprocessing complete!

train:
	@echo Training models...
	python -m src.churn.train
	@echo Training complete!

evaluate:
	@echo Evaluating on test set...
	python -m src.churn.evaluate
	@echo Evaluation complete!

explain:
	@echo Generating interpretation and business optimization...
	python -m src.churn.explain
	@echo Interpretation complete!

all: preprocess train evaluate explain
	@echo ""
	@echo ========================================
	@echo   FULL PIPELINE COMPLETE
	@echo ========================================
	@echo ""
	@echo Results:
	@echo   - Models saved to: models/
	@echo   - Reports in: reports/
	@echo   - Visualizations in: reports/figures/
	@echo ""

clean:
	@echo Cleaning generated files...
	@if exist data\processed\*.npy del /Q data\processed\*.npy
	@if exist models\*.pkl del /Q models\*.pkl
	@if exist reports\*.csv del /Q reports\*.csv
	@if exist reports\figures\*.png del /Q reports\figures\*.png
	@echo Clean complete!
