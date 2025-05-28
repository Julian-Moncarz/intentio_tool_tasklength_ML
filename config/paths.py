#!/usr/bin/env python3
"""
Configuration file for project paths.
This makes it easy to manage file locations across all scripts.
"""

import os
from pathlib import Path

# Get the project root directory (parent of config/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# Data files
LOGS_CSV = DATA_DIR / "logs.csv"
EMBEDDINGS_PKL = DATA_DIR / "embeddings_data.pkl"
EMBEDDINGS_JSON = DATA_DIR / "embeddings_data.json"
EMBEDDINGS_SUMMARY_CSV = DATA_DIR / "embeddings_summary.csv"

# Model files
RANDOM_FOREST_MODEL = MODELS_DIR / "random_forest_model.pkl"
BEST_MODEL = MODELS_DIR / "best_model.pkl"
ALL_MODEL_RESULTS = MODELS_DIR / "all_model_results.pkl"

# Result files
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison_results.csv"
COMPREHENSIVE_COMPARISON_PNG = RESULTS_DIR / "comprehensive_model_comparison.png"
BEST_MODEL_PREDICTIONS_PNG = RESULTS_DIR / "best_model_predictions.png"
RANDOM_FOREST_RESULTS_PNG = RESULTS_DIR / "random_forest_results.png"

def ensure_directories():
    """Create all necessary directories if they don't exist"""
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(exist_ok=True)

if __name__ == "__main__":
    # Print all paths for debugging
    print("Project paths configuration:")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"SRC_DIR: {SRC_DIR}")
