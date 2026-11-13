"""
Real-Time Fraud Detection Engine.

Wraps the trained ensemble model + feature pipeline into a low-latency
prediction service. Supports single-click and batch prediction, with
built-in sliding-window statistics for real-time behavioral features.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    FRAUD_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    LGBM_MODEL_PATH,
    XGB_MODEL_PATH,
    ENSEMBLE_MODEL_PATH,
    CLICK_STATS_PATH,
    ENSEMBLE_WEIGHTS,
)
from src.feature_engineering import FeatureEngineer


class RealTimeFraudDetector:
    """
    Production-grade real-time fraud detection engine.

    Features:
      - Sub-millisecond single-click prediction
      - Sliding-window IP behavior tracking
      - Batch prediction for bulk processing
      - Fraud risk categorization (low / medium / high)
      - Session-level anomaly aggregation
    """

    def __init__(self, window_seconds: int = 300):
        """
        Parameters
        ----------
        window_seconds : int
            Sliding window size for real-time behavioral features (default: 5 min)
        """
        self.feature_engineer = FeatureEngineer()
        self.lgbm_model = None
        self.xgb_model = None
        self.ensemble_meta = None
        self.feature_names: list = []

        # ── Real-time sliding-window state ──
        self.window_seconds = window_seconds
        self.ip_click_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self.ip_app_counter: Dict[Tuple[int, int], int] = defaultdict(int)

        # ── Statistics ──
        self.total_predictions = 0
        self.total_fraud_flagged = 0
        self.latency_history: deque = deque(maxlen=10000)
        self.fraud_threshold = float(FRAUD_THRESHOLD)

        self._loaded = False

    # ─────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────
    def load(self):
        """Load trained models and feature pipeline from disk."""
        print("[FraudDetector] Loading models...")

        self.lgbm_model = joblib.load(LGBM_MODEL_PATH)
        self.xgb_model = joblib.load(XGB_MODEL_PATH)
        self.ensemble_meta = joblib.load(ENSEMBLE_MODEL_PATH)
        self.feature_names = self.ensemble_meta["feature_names"]
        self.fraud_threshold = float(self.ensemble_meta.get("optimal_threshold", FRAUD_THRESHOLD))

        self.feature_engineer.load(CLICK_STATS_PATH)
        self._loaded = True
        print(f"[FraudDetector] Ready for predictions (threshold: {self.fraud_threshold:.4f})")

    # ─────────────────────────────────────────────────────────────
    # Single-Click Prediction
    # ─────────────────────────────────────────────────────────────
    def predict_click(self, click: dict, track_realtime: bool = False) -> dict:
        """
        Predict fraud probability for a single click event.

        Parameters
        ----------
        click : dict
            Raw click event with keys: ip, app, device, os, channel, click_time
        track_realtime : bool
            If True, update sliding-window state for real-time velocity & burst detection.
            If False (default for single-click inspection), evaluate deterministically.

        Returns
        -------
        dict with:
            fraud_probability, is_fraud, risk_level, latency_ms,
            real_time_features (IP velocity, burst detection, etc.)
        """
        assert self._loaded, "Call .load() before prediction"
        t0 = time.perf_counter()

        # 1. Update sliding-window state (only when tracking is enabled)
        if track_realtime:
            rt_features = self._update_realtime_state(click)
        else:
            rt_features = {
                "rt_ip_click_velocity": 1,
                "rt_ip_unique_apps": 1,
                "rt_ip_unique_channels": 1,
                "rt_ip_unique_devices": 1,
                "rt_seconds_since_last_click": -1.0,
                "rt_burst_count_1s": 1,
                "rt_app_concentration": 1.0,
            }

        # 2. Feature engineering
        df = self.feature_engineer.transform_single(click)

        # 3. Dynamically augment statistical features with real-time sliding window stats (if tracking)
        if track_realtime:
            if "clicks_ip" in df.columns:
                df["clicks_ip"] = max(int(df["clicks_ip"].iloc[0]), rt_features["rt_ip_click_velocity"])
            if "clicks_ip_device" in df.columns:
                df["clicks_ip_device"] = max(int(df["clicks_ip_device"].iloc[0]), rt_features["rt_ip_click_velocity"])
            if "unique_channel_by_ip" in df.columns:
                df["unique_channel_by_ip"] = max(int(df["unique_channel_by_ip"].iloc[0]), rt_features["rt_ip_unique_channels"])
            if "unique_app_by_ip" in df.columns:
                df["unique_app_by_ip"] = max(int(df["unique_app_by_ip"].iloc[0]), rt_features["rt_ip_unique_apps"])
            if "unique_device_by_ip" in df.columns:
                df["unique_device_by_ip"] = max(int(df["unique_device_by_ip"].iloc[0]), rt_features["rt_ip_unique_devices"])

            if rt_features["rt_seconds_since_last_click"] >= 0:
                for delta_col in [c for c in df.columns if c.startswith("prev_click_")]:
                    df[delta_col] = float(rt_features["rt_seconds_since_last_click"])

        # Inject real-time features for reporting
        for k, v in rt_features.items():
            df[k] = v

        # 4. Extract feature matrix
        feature_cols = [c for c in self.feature_names if c in df.columns]
        X = df[feature_cols].values.astype(np.float32)

        # Pad missing features with zeros
        if X.shape[1] < len(self.feature_names):
            X_padded = np.zeros((1, len(self.feature_names)), dtype=np.float32)
            for i, fn in enumerate(self.feature_names):
                if fn in feature_cols:
                    col_idx = feature_cols.index(fn)
                    X_padded[0, i] = X[0, col_idx]
            X = X_padded

        # 5. Ensemble prediction
        p_lgbm = float(self.lgbm_model.predict_proba(X)[:, 1][0])
        p_xgb = float(self.xgb_model.predict_proba(X)[:, 1][0])
        fraud_prob = float(
            ENSEMBLE_WEIGHTS["lgbm"] * p_lgbm
            + ENSEMBLE_WEIGHTS["xgb"] * p_xgb
        )

        # Fast-path anomaly heuristic: bot burst / flood detection in sliding window (only when tracking)
        if track_realtime:
            is_bot_burst = (
                rt_features["rt_burst_count_1s"] >= 4
                or (rt_features["rt_ip_click_velocity"] >= 8 and 0 <= rt_features["rt_seconds_since_last_click"] < 0.5)
            )
            if is_bot_burst:
                fraud_prob = max(fraud_prob, 0.96)

        # 6. Risk categorization
        is_fraud = fraud_prob >= self.fraud_threshold
        risk_level = self._categorize_risk(fraud_prob, self.fraud_threshold)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Track stats
        self.total_predictions += 1
        if is_fraud:
            self.total_fraud_flagged += 1
        self.latency_history.append(latency_ms)

        return {
            "fraud_probability": round(fraud_prob, 6),
            "lgbm_probability": round(float(p_lgbm), 6),
            "xgb_probability": round(float(p_xgb), 6),
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "latency_ms": round(latency_ms, 3),
            "real_time_features": rt_features,
            "click": click,
        }

    # ─────────────────────────────────────────────────────────────
    # Batch Prediction
    # ─────────────────────────────────────────────────────────────
    def predict_batch(self, clicks: List[dict], track_realtime: bool = True) -> List[dict]:
        """Predict fraud for a batch of clicks."""
        return [self.predict_click(click, track_realtime=track_realtime) for click in clicks]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction on a DataFrame. Returns the DataFrame with
        fraud_probability, is_fraud, and risk_level columns added.
        """
        assert self._loaded, "Call .load() before prediction"

        df_feat = self.feature_engineer.transform(df.copy())
        feature_cols = [c for c in self.feature_names if c in df_feat.columns]
        X = df_feat[feature_cols].values.astype(np.float32)

        # Pad missing features
        if X.shape[1] < len(self.feature_names):
            X_padded = np.zeros((len(df), len(self.feature_names)), dtype=np.float32)
            for i, fn in enumerate(self.feature_names):
                if fn in feature_cols:
                    col_idx = feature_cols.index(fn)
                    X_padded[:, i] = X[:, col_idx]
            X = X_padded

        p_lgbm = self.lgbm_model.predict_proba(X)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        fraud_prob = ENSEMBLE_WEIGHTS["lgbm"] * p_lgbm + ENSEMBLE_WEIGHTS["xgb"] * p_xgb

        df["fraud_probability"] = fraud_prob.round(6)
        df["is_fraud"] = fraud_prob >= FRAUD_THRESHOLD
        df["risk_level"] = [self._categorize_risk(p) for p in fraud_prob]

        return df

    # ─────────────────────────────────────────────────────────────
    # Real-Time Sliding Window Features
    # ─────────────────────────────────────────────────────────────
    def _update_realtime_state(self, click: dict) -> dict:
        """
        Maintain per-IP sliding window and compute real-time features:
          - IP click velocity (clicks in last N seconds)
          - Burst detection (rapid successive clicks)
          - App diversity in window
          - Channel concentration
        """
        ip = click.get("ip", 0)
        click_time = click.get("click_time")
        if isinstance(click_time, str):
            click_time = pd.Timestamp(click_time)
        elif isinstance(click_time, datetime):
            click_time = pd.Timestamp(click_time)
        elif click_time is None:
            click_time = pd.Timestamp.utcnow()

        # Normalize to tz-naive UTC for reliable timestamp comparison
        if hasattr(click_time, "tzinfo") and click_time.tzinfo is not None:
            click_time = click_time.tz_convert("UTC").tz_localize(None)

        # Record this click
        self.ip_click_history[ip].append({
            "time": click_time,
            "app": click.get("app", 0),
            "device": click.get("device", 0),
            "channel": click.get("channel", 0),
        })

        # Prune expired entries
        cutoff = click_time - timedelta(seconds=self.window_seconds)
        history = self.ip_click_history[ip]
        while history and history[0]["time"] < cutoff:
            history.popleft()

        # Compute features
        n_clicks = len(history)
        apps_in_window = set(h["app"] for h in history)
        channels_in_window = set(h["channel"] for h in history)
        devices_in_window = set(h["device"] for h in history)

        # Time since last click from this IP
        if n_clicks >= 2:
            last_delta = (history[-1]["time"] - history[-2]["time"]).total_seconds()
        else:
            last_delta = -1.0

        # Burst: clicks within 1 second
        burst_count = sum(
            1 for h in history
            if (click_time - h["time"]).total_seconds() <= 1
        )

        return {
            "rt_ip_click_velocity": n_clicks,
            "rt_ip_unique_apps": len(apps_in_window),
            "rt_ip_unique_channels": len(channels_in_window),
            "rt_ip_unique_devices": len(devices_in_window),
            "rt_seconds_since_last_click": round(last_delta, 3),
            "rt_burst_count_1s": burst_count,
            "rt_app_concentration": (
                n_clicks / max(len(apps_in_window), 1)
            ),
        }

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _categorize_risk(prob: float, threshold: float = FRAUD_THRESHOLD) -> str:
        high_thresh = min(0.90, max(threshold + 0.15, 0.70))
        if prob >= high_thresh:
            return "HIGH"
        elif prob >= threshold:
            return "MEDIUM"
        elif prob >= threshold * 0.4:
            return "LOW"
        else:
            return "SAFE"

    def get_engine_stats(self) -> dict:
        """Return engine-level operational statistics."""
        latencies = list(self.latency_history) if self.latency_history else [0]
        return {
            "total_predictions": self.total_predictions,
            "total_fraud_flagged": self.total_fraud_flagged,
            "fraud_rate": (
                self.total_fraud_flagged / max(self.total_predictions, 1)
            ),
            "avg_latency_ms": round(np.mean(latencies), 3),
            "p50_latency_ms": round(np.percentile(latencies, 50), 3),
            "p95_latency_ms": round(np.percentile(latencies, 95), 3),
            "p99_latency_ms": round(np.percentile(latencies, 99), 3),
            "tracked_ips": len(self.ip_click_history),
        }
