"""
End-to-End Training Pipeline.

Orchestrates:
  1. Data loading (or synthetic generation)
  2. Exploratory Data Analysis
  3. Feature Engineering
  4. Model Training (LightGBM + XGBoost)
  5. Ensemble Evaluation
  6. Model Persistence
  7. Visualization of Results
"""

import os
import sys
import time
import gc
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    RAW_TRAIN_PATH,
    SAMPLE_TRAIN_PATH,
    TRAIN_SAMPLE_SIZE,
    VALIDATION_FRACTION,
    DATA_DIR,
    MODEL_DIR,
    DTYPES,
)
from src.data_generator import generate_talkingdata_sample
from src.feature_engineering import FeatureEngineer
from src.model_training import FraudModelTrainer
from src.visualization import (
    generate_full_eda,
    plot_model_performance,
    plot_feature_importance,
    plot_threshold_analysis,
)


def load_data(sample_size: int = TRAIN_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Load training data. Priorities:
      1. Full Kaggle dataset (train.csv)
      2. Pre-generated sample (train_sample.csv)
      3. Auto-generate synthetic data
    """
    if os.path.exists(RAW_TRAIN_PATH):
        print(f"[Data] Loading Kaggle dataset from {RAW_TRAIN_PATH} (first {sample_size:,} rows)")
        df = pd.read_csv(
            RAW_TRAIN_PATH,
            dtype=DTYPES,
            parse_dates=["click_time", "attributed_time"],
            nrows=sample_size,
        )
    elif os.path.exists(SAMPLE_TRAIN_PATH):
        print(f"[Data] Loading sample from {SAMPLE_TRAIN_PATH}")
        df = pd.read_csv(
            SAMPLE_TRAIN_PATH,
            dtype=DTYPES,
            parse_dates=["click_time"],
        )
    else:
        print("[Data] No dataset found — generating synthetic data...")
        df = generate_talkingdata_sample(n_rows=500_000, save=True)

    print(f"[Data] Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"[Data] Fraud rate: {df['is_attributed'].mean():.4%}")
    print(f"[Data] Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def main(sample_size: int = TRAIN_SAMPLE_SIZE):
    """Run the complete training pipeline."""
    pipeline_start = time.time()

    print("\n" + "═" * 70)
    print("  🛡️  MOBILE AD CLICK FRAUD DETECTION — TRAINING PIPELINE")
    print("═" * 70 + "\n")

    # ──────────────────────────────────────────────────────────
    # Step 1: Load Data
    # ──────────────────────────────────────────────────────────
    print("━" * 50)
    print("  STEP 1/6 • Data Loading")
    print("━" * 50)
    df = load_data(sample_size=sample_size)
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nNull counts:\n{df.isnull().sum()}")

    # ──────────────────────────────────────────────────────────
    # Step 2: EDA
    # ──────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("  STEP 2/6 • Exploratory Data Analysis")
    print("━" * 50)
    generate_full_eda(df)

    # ──────────────────────────────────────────────────────────
    # Step 3: Chronological Train/Val Split (Before Feature Fit)
    # ──────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("  STEP 3/6 • Chronological Train/Val Split (Leak-Free)")
    print("━" * 50)
    # Ensure chronological order
    df = df.sort_values("click_time").reset_index(drop=True)
    split_idx = int(len(df) * (1 - VALIDATION_FRACTION))
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    del df
    gc.collect()

    print(f"[Split] Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")
    print(f"[Split] Train fraud rate: {train_df['is_attributed'].mean():.4%} | Val fraud rate: {val_df['is_attributed'].mean():.4%}")

    # ──────────────────────────────────────────────────────────
    # Step 4: Feature Engineering (Fitted ONLY on Historical Train)
    # ──────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("  STEP 4/6 • Feature Engineering (Strictly Historical)")
    print("━" * 50)
    fe = FeatureEngineer()
    train_df = fe.fit_transform(train_df)
    feature_cols = fe.feature_names

    print(f"\nEngineered {len(feature_cols)} features (fitted on train only):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    fe.save()

    # Transform validation using ONLY training lookups (out-of-vocabulary defaults)
    val_df = fe.transform(val_df)

    trainer = FraudModelTrainer()
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["is_attributed"].values.astype(np.uint8)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["is_attributed"].values.astype(np.uint8)

    del train_df, val_df
    gc.collect()

    # ──────────────────────────────────────────────────────────
    # Step 5: Model Training
    # ──────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("  STEP 5/6 • Model Training")
    print("━" * 50)

    trainer.train_lightgbm(X_train, X_val, y_train, y_val, feature_cols)
    trainer.train_xgboost(X_train, X_val, y_train, y_val, feature_cols)

    # ──────────────────────────────────────────────────────────
    # Step 6: Ensemble Evaluation
    # ──────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("  STEP 6/6 • Ensemble Evaluation & Visualization")
    print("━" * 50)

    metrics = trainer.evaluate_ensemble(X_val, y_val)

    # Feature importance
    importance_df = trainer.get_feature_importance(top_n=20)
    print(f"\nTop 20 Feature Importance:\n{importance_df.to_string(index=False)}")

    # Save models
    trainer.save_models()

    # ── Visualizations ──
    y_proba = trainer.ensemble_predict(X_val)
    plot_model_performance(y_val, y_proba, model_name="LightGBM+XGBoost Ensemble")
    plot_feature_importance(importance_df)
    plot_threshold_analysis(y_val, y_proba)

    # ── Summary ──
    elapsed = time.time() - pipeline_start
    print("\n" + "═" * 70)
    print("  ✅  TRAINING PIPELINE COMPLETE")
    print("═" * 70)
    print(f"""
    Total time:          {elapsed:.1f}s
    Features:            {len(feature_cols)}
    Training rows:       {X_train.shape[0]:,}
    Validation rows:     {X_val.shape[0]:,}

    ┌─────────────────────────────────────────┐
    │  MODEL PERFORMANCE (Validation Set)     │
    ├─────────────────────────────────────────┤
    │  LightGBM AUC-ROC:   {trainer.training_metrics['lgbm']['auc_roc']:.6f}          │
    │  XGBoost  AUC-ROC:   {trainer.training_metrics['xgb']['auc_roc']:.6f}          │
    │  Ensemble AUC-ROC:   {metrics['auc_roc']:.6f}          │
    │  Ensemble Avg Prec:  {metrics['avg_precision']:.6f}          │
    │  Optimal Threshold:  {metrics['optimal_threshold']:.4f}            │
    │  Optimal F1:         {metrics['optimal_f1']:.6f}          │
    └─────────────────────────────────────────┘

    Confusion Matrix:
    ┌───────────┬────────────┬───────────┐
    │           │ Pred Legit │ Pred Fraud│
    ├───────────┼────────────┼───────────┤
    │ True Legit│ {metrics['tn']:>10,} │ {metrics['fp']:>9,} │
    │ True Fraud│ {metrics['fn']:>10,} │ {metrics['tp']:>9,} │
    └───────────┴────────────┴───────────┘

    Models saved to:     models/
    Plots saved to:      plots/
    """)

    return trainer, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Mobile Ad Click Fraud Detection Pipeline")
    parser.add_argument("--rows", type=int, default=TRAIN_SAMPLE_SIZE, help="Number of rows to load from train.csv")
    args = parser.parse_args()
    main(sample_size=args.rows)
