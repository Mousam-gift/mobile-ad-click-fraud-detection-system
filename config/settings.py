"""
Centralized Configuration for Mobile Ad Click Fraud Detection Engine.
All hyperparameters, file paths, feature definitions, and thresholds.
"""

import os

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

RAW_TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
RAW_TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_TRAIN_PATH = os.path.join(DATA_DIR, "train_sample.csv")

# Saved artefacts
FEATURE_PIPELINE_PATH = os.path.join(MODEL_DIR, "feature_pipeline.joblib")
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_fraud_model.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_fraud_model.joblib")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
CLICK_STATS_PATH = os.path.join(MODEL_DIR, "click_stats.joblib")
TRAINING_METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")

# ──────────────────────────────────────────────────────────────────────
# Data Schema  –  TalkingData columns
# ──────────────────────────────────────────────────────────────────────
RAW_COLUMNS = [
    "ip", "app", "device", "os", "channel",
    "click_time", "attributed_time", "is_attributed",
]

DTYPES = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
    "is_attributed": "uint8",
}

# ──────────────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────────────
CLICK_COUNT_GROUPKEYS = [
    ["ip"],
    ["ip", "app"],
    ["ip", "app", "os"],
    ["ip", "device"],
    ["ip", "channel"],
    ["app", "channel"],
]

TIME_DELTA_GROUPKEYS = [
    ["ip", "app", "device", "os"],
    ["ip", "os", "device"],
    ["ip", "os", "device", "app"],
]

UNIQUE_COUNT_SPECS = [
    # (group_cols, target_col)
    ("ip", "channel"),
    ("ip", "app"),
    ("ip", "device"),
    ("ip", "os"),
    ("app", "channel"),
    ("ip_app", "os"),       # combined key
    ("ip_device_os", "app"),
]

HOUR_BINS = [0, 6, 12, 18, 24]
HOUR_LABELS = ["night", "morning", "afternoon", "evening"]

# ──────────────────────────────────────────────────────────────────────
# Model Hyperparameters
# ──────────────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,  # Unweighted for calibrated probabilities & stable gradient boosting
    "n_estimators": 600,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,  # Unweighted for calibrated probabilities
    "n_estimators": 600,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",
}

ENSEMBLE_WEIGHTS = {"lgbm": 0.6, "xgb": 0.4}

# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
TRAIN_SAMPLE_SIZE = 2_000_000        # rows to use from train.csv (out of ~185M)
VALIDATION_FRACTION = 0.1            # last 10% chronologically
EARLY_STOPPING_ROUNDS = 40
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────
# Serving / Real-Time
# ──────────────────────────────────────────────────────────────────────
FRAUD_THRESHOLD = 0.35      # Calibrated decision threshold
HIGH_RISK_THRESHOLD = 0.70  # High-confidence fraud
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4
MAX_BATCH_SIZE = 1000
REQUEST_TIMEOUT_SEC = 5
