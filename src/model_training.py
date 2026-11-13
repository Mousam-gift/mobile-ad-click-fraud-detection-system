"""
Model Training Pipeline — LightGBM + XGBoost Ensemble.

Trains binary classifiers for ad click fraud detection with:
  - Chronological train/val split (prevents look-ahead leakage)
  - Class-imbalance handling via scale_pos_weight
  - Early stopping on AUC-ROC
  - Ensemble via weighted probability averaging
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from typing import Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    LGBM_PARAMS,
    XGB_PARAMS,
    ENSEMBLE_WEIGHTS,
    VALIDATION_FRACTION,
    EARLY_STOPPING_ROUNDS,
    LGBM_MODEL_PATH,
    XGB_MODEL_PATH,
    ENSEMBLE_MODEL_PATH,
    TRAINING_METRICS_PATH,
    FRAUD_THRESHOLD,
    MODEL_DIR,
)


class FraudModelTrainer:
    """
    Trains and evaluates LightGBM + XGBoost ensemble for fraud detection.
    """

    def __init__(self):
        self.lgbm_model = None
        self.xgb_model = None
        self.feature_names: list = []
        self.training_metrics: dict = {}

    # ─────────────────────────────────────────────────────────────
    # Data Splitting
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = "is_attributed",
        val_frac: float = VALIDATION_FRACTION,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Time-ordered split to prevent data leakage.
        Last `val_frac` of rows become the validation set.
        """
        split_idx = int(len(df) * (1 - val_frac))

        X_train = df.iloc[:split_idx][feature_cols].values.astype(np.float32)
        y_train = df.iloc[:split_idx][target_col].values.astype(np.uint8)
        X_val = df.iloc[split_idx:][feature_cols].values.astype(np.float32)
        y_val = df.iloc[split_idx:][target_col].values.astype(np.uint8)

        print(f"[Split] Train: {X_train.shape[0]:,} rows | Val: {X_val.shape[0]:,} rows")
        print(f"[Split] Train fraud rate: {y_train.mean():.4%} | Val fraud rate: {y_val.mean():.4%}")

        return X_train, X_val, y_train, y_val

    # ─────────────────────────────────────────────────────────────
    # LightGBM
    # ─────────────────────────────────────────────────────────────
    def train_lightgbm(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: list,
    ) -> lgb.LGBMClassifier:
        """Train LightGBM with early stopping."""
        print("\n" + "=" * 60)
        print("  Training LightGBM")
        print("=" * 60)

        t0 = time.time()

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(100),
            ],
        )

        elapsed = time.time() - t0
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        ap = average_precision_score(y_val, y_pred_proba)

        print(f"\n[LightGBM] Training time: {elapsed:.1f}s")
        print(f"[LightGBM] Val AUC-ROC:  {auc:.6f}")
        print(f"[LightGBM] Val AP (PR):   {ap:.6f}")
        print(f"[LightGBM] Best iteration: {model.best_iteration_}")

        self.lgbm_model = model
        self.feature_names = feature_names
        self.training_metrics["lgbm"] = {
            "auc_roc": float(auc),
            "avg_precision": float(ap),
            "training_time_sec": round(elapsed, 2),
            "best_iteration": model.best_iteration_,
            "n_estimators": model.n_estimators,
        }

        return model

    # ─────────────────────────────────────────────────────────────
    # XGBoost
    # ─────────────────────────────────────────────────────────────
    def train_xgboost(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: list,
    ) -> xgb.XGBClassifier:
        """Train XGBoost with early stopping."""
        print("\n" + "=" * 60)
        print("  Training XGBoost")
        print("=" * 60)

        t0 = time.time()

        params = {**XGB_PARAMS}
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        elapsed = time.time() - t0
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        ap = average_precision_score(y_val, y_pred_proba)

        try:
            best_iter = int(model.best_iteration)
        except (AttributeError, TypeError):
            best_iter = model.n_estimators

        print(f"\n[XGBoost] Training time: {elapsed:.1f}s")
        print(f"[XGBoost] Val AUC-ROC:  {auc:.6f}")
        print(f"[XGBoost] Val AP (PR):   {ap:.6f}")
        print(f"[XGBoost] Best iteration: {best_iter}")

        self.xgb_model = model
        self.training_metrics["xgb"] = {
            "auc_roc": float(auc),
            "avg_precision": float(ap),
            "training_time_sec": round(elapsed, 2),
            "best_iteration": best_iter,
            "n_estimators": model.n_estimators,
        }

        return model

    # ─────────────────────────────────────────────────────────────
    # Ensemble
    # ─────────────────────────────────────────────────────────────
    def ensemble_predict(
        self,
        X: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Weighted ensemble of LightGBM + XGBoost probabilities."""
        weights = weights or ENSEMBLE_WEIGHTS

        proba_lgbm = self.lgbm_model.predict_proba(X)[:, 1]
        proba_xgb = self.xgb_model.predict_proba(X)[:, 1]

        ensemble_proba = (
            weights["lgbm"] * proba_lgbm + weights["xgb"] * proba_xgb
        )
        return ensemble_proba

    def evaluate_ensemble(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        threshold: float = FRAUD_THRESHOLD,
    ) -> dict:
        """Full evaluation of ensemble model on validation set."""
        print("\n" + "=" * 60)
        print("  Ensemble Evaluation")
        print("=" * 60)

        y_proba = self.ensemble_predict(X_val)
        y_pred = (y_proba >= threshold).astype(int)

        auc = roc_auc_score(y_val, y_proba)
        ap = average_precision_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        # Find optimal threshold for F1
        precisions, recalls, thresholds_pr = precision_recall_curve(y_val, y_proba)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds_pr[optimal_idx]) if optimal_idx < len(thresholds_pr) else threshold

        print(f"[Ensemble] AUC-ROC:           {auc:.6f}")
        print(f"[Ensemble] Avg Precision (PR): {ap:.6f}")
        print(f"[Ensemble] F1 @ {threshold}:     {f1:.6f}")
        print(f"[Ensemble] Optimal threshold:  {optimal_threshold:.4f}")
        print(f"[Ensemble] Optimal F1:         {f1_scores[optimal_idx]:.6f}")
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nClassification Report @ threshold={threshold}:")
        print(classification_report(y_val, y_pred, target_names=["Legitimate", "Fraud"]))

        metrics = {
            "auc_roc": float(auc),
            "avg_precision": float(ap),
            "f1_at_threshold": float(f1),
            "threshold": threshold,
            "optimal_threshold": optimal_threshold,
            "optimal_f1": float(f1_scores[optimal_idx]),
            "confusion_matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
        self.training_metrics["ensemble"] = metrics
        return metrics

    # ─────────────────────────────────────────────────────────────
    # Feature Importance
    # ─────────────────────────────────────────────────────────────
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Combined feature importance from both models."""
        lgbm_imp = self.lgbm_model.feature_importances_
        xgb_imp = self.xgb_model.feature_importances_

        # Normalize to [0,1]
        lgbm_norm = lgbm_imp / (lgbm_imp.sum() + 1e-8)
        xgb_norm = xgb_imp / (xgb_imp.sum() + 1e-8)

        combined = (
            ENSEMBLE_WEIGHTS["lgbm"] * lgbm_norm
            + ENSEMBLE_WEIGHTS["xgb"] * xgb_norm
        )

        importance_df = pd.DataFrame({
            "feature": self.feature_names[:len(combined)],
            "lgbm_importance": lgbm_norm[:len(self.feature_names)],
            "xgb_importance": xgb_norm[:len(self.feature_names)],
            "ensemble_importance": combined,
        }).sort_values("ensemble_importance", ascending=False)

        return importance_df.head(top_n)

    # ─────────────────────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────────────────────
    def save_models(self):
        """Persist trained models and metrics."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(self.lgbm_model, LGBM_MODEL_PATH)
        joblib.dump(self.xgb_model, XGB_MODEL_PATH)
        joblib.dump({
            "lgbm_path": LGBM_MODEL_PATH,
            "xgb_path": XGB_MODEL_PATH,
            "weights": ENSEMBLE_WEIGHTS,
            "feature_names": self.feature_names,
            "optimal_threshold": float(self.training_metrics.get("ensemble", {}).get("optimal_threshold", FRAUD_THRESHOLD)),
        }, ENSEMBLE_MODEL_PATH)

        with open(TRAINING_METRICS_PATH, "w") as f:
            json.dump(self.training_metrics, f, indent=2)

        print(f"\n[Save] LightGBM → {LGBM_MODEL_PATH}")
        print(f"[Save] XGBoost  → {XGB_MODEL_PATH}")
        print(f"[Save] Ensemble → {ENSEMBLE_MODEL_PATH}")
        print(f"[Save] Metrics  → {TRAINING_METRICS_PATH}")

    def load_models(self):
        """Load pre-trained models."""
        self.lgbm_model = joblib.load(LGBM_MODEL_PATH)
        self.xgb_model = joblib.load(XGB_MODEL_PATH)

        ensemble_meta = joblib.load(ENSEMBLE_MODEL_PATH)
        self.feature_names = ensemble_meta["feature_names"]

        if os.path.exists(TRAINING_METRICS_PATH):
            with open(TRAINING_METRICS_PATH) as f:
                self.training_metrics = json.load(f)

        print("[Load] Models loaded successfully")
