"""
Feature Engineering Pipeline for Ad Click Fraud Detection.

Extracts temporal, behavioral, and statistical features from raw click logs.
Designed for both batch training and real-time single-click inference.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    CLICK_COUNT_GROUPKEYS,
    TIME_DELTA_GROUPKEYS,
    HOUR_BINS,
    HOUR_LABELS,
    CLICK_STATS_PATH,
    MODEL_DIR,
)


class FeatureEngineer:
    """
    Stateful feature engineering pipeline.

    Training mode:  fit_transform(df) — computes lookup tables and transforms.
    Inference mode:  transform(df)    — uses saved lookup tables.
    """

    def __init__(self):
        self.click_stats: dict = {}    # group → count lookup
        self.unique_stats: dict = {}   # group → nunique lookup
        self.fitted = False
        self.feature_names: list = []

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return engineered features."""
        print("[FeatureEngineer] fit_transform on {:,} rows".format(len(df)))
        df = self._parse_time(df)
        df = self._add_time_features(df)
        df = self._add_click_counts(df, fit=True)
        df = self._add_time_deltas(df)
        df = self._add_unique_counts(df, fit=True)
        df = self._add_interaction_features(df)
        df = self._add_rate_features(df)
        self.fitted = True
        self.feature_names = self._get_feature_columns(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted lookup tables."""
        df = self._parse_time(df)
        df = self._add_time_features(df)
        df = self._add_click_counts(df, fit=False)
        df = self._add_time_deltas(df)
        df = self._add_unique_counts(df, fit=False)
        df = self._add_interaction_features(df)
        df = self._add_rate_features(df)
        return df

    def transform_single(self, click: dict) -> pd.DataFrame:
        """Transform a single click event (real-time inference)."""
        df = pd.DataFrame([click])
        return self.transform(df)

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the final feature matrix as a numpy array."""
        cols = [c for c in self.feature_names if c in df.columns]
        return df[cols].values.astype(np.float32)

    def save(self, path: Optional[str] = None):
        """Persist the fitted pipeline to disk."""
        path = path or CLICK_STATS_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "click_stats": self.click_stats,
            "unique_stats": self.unique_stats,
            "feature_names": self.feature_names,
        }, path)
        print(f"[FeatureEngineer] Saved stats → {path}")

    def load(self, path: Optional[str] = None):
        """Load a previously fitted pipeline."""
        path = path or CLICK_STATS_PATH
        data = joblib.load(path)
        self.click_stats = data["click_stats"]
        self.unique_stats = data["unique_stats"]
        self.feature_names = data["feature_names"]
        self.fitted = True
        print(f"[FeatureEngineer] Loaded stats ← {path}")

    # ─────────────────────────────────────────────────────────────
    # Time Features
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_time(df: pd.DataFrame) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df["click_time"]):
            df["click_time"] = pd.to_datetime(df["click_time"], utc=True).dt.tz_localize(None)
        elif hasattr(df["click_time"].dt, "tz") and df["click_time"].dt.tz is not None:
            df["click_time"] = df["click_time"].dt.tz_convert("UTC").dt.tz_localize(None)
        return df

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        ct = df["click_time"]
        df["hour"] = ct.dt.hour.astype(np.uint8)
        df["minute"] = ct.dt.minute.astype(np.uint8)
        df["second"] = ct.dt.second.astype(np.uint8)
        df["day_of_week"] = ct.dt.dayofweek.astype(np.uint8)
        df["day"] = ct.dt.day.astype(np.uint8)

        # Cyclical encoding for hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)

        # Time-of-day bin
        df["time_of_day"] = pd.cut(
            df["hour"], bins=HOUR_BINS, labels=HOUR_LABELS, right=False, ordered=False,
        ).cat.codes.astype(np.int8)

        return df

    # ─────────────────────────────────────────────────────────────
    # Click Counts (frequency-based)
    # ─────────────────────────────────────────────────────────────
    def _add_click_counts(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        for group_cols in CLICK_COUNT_GROUPKEYS:
            col_name = "clicks_" + "_".join(group_cols)

            if fit:
                counts = df.groupby(group_cols).size().reset_index(name=col_name)
                self.click_stats[col_name] = counts
                df = df.merge(counts, on=group_cols, how="left")
            else:
                if col_name in self.click_stats:
                    df = df.merge(self.click_stats[col_name], on=group_cols, how="left")
                    df[col_name] = df[col_name].fillna(1).astype(np.uint32)
                else:
                    df[col_name] = 1

        return df

    # ─────────────────────────────────────────────────────────────
    # Time Deltas (Causal backward-looking: seconds since previous click)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _add_time_deltas(df: pd.DataFrame) -> pd.DataFrame:
        """
        Causal backward-looking time deltas (seconds since previous click).
        Leak-free: computable online in real-time inference without look-ahead.
        """
        for group_cols in TIME_DELTA_GROUPKEYS:
            prev_col = "prev_click_" + "_".join(group_cols)
            df[prev_col] = (
                df.groupby(group_cols)["click_time"]
                .diff()
                .dt.total_seconds()
                .fillna(-1)
                .astype(np.float32)
            )

        return df

    # ─────────────────────────────────────────────────────────────
    # Unique Value Counts (diversity features)
    # ─────────────────────────────────────────────────────────────
    def _add_unique_counts(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        unique_specs = [
            (["ip"], "channel"),
            (["ip"], "app"),
            (["ip"], "device"),
            (["ip"], "os"),
            (["app"], "channel"),
            (["ip", "app"], "os"),
            (["ip", "device", "os"], "app"),
        ]

        for group_cols, target_col in unique_specs:
            col_name = f"unique_{target_col}_by_{'_'.join(group_cols)}"

            if fit:
                nunique = (
                    df.groupby(group_cols)[target_col]
                    .nunique()
                    .reset_index(name=col_name)
                )
                self.unique_stats[col_name] = nunique
                df = df.merge(nunique, on=group_cols, how="left")
            else:
                if col_name in self.unique_stats:
                    df = df.merge(self.unique_stats[col_name], on=group_cols, how="left")
                    df[col_name] = df[col_name].fillna(1).astype(np.uint32)
                else:
                    df[col_name] = 1

        return df

    # ─────────────────────────────────────────────────────────────
    # Interaction Features
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        df["ip_app"] = (df["ip"].astype(np.uint64) * 1000 + df["app"]).astype(np.uint64)
        df["ip_device_os"] = (
            df["ip"].astype(np.uint64) * 10000
            + df["device"].astype(np.uint64) * 100
            + df["os"]
        ).astype(np.uint64)
        df["app_channel"] = (df["app"].astype(np.uint32) * 1000 + df["channel"]).astype(np.uint32)

        # Hashed interaction (avoid large cardinality)
        df["ip_app_hash"] = (df["ip_app"] % 100_000).astype(np.uint32)
        df["ip_device_os_hash"] = (df["ip_device_os"] % 100_000).astype(np.uint32)

        return df

    # ─────────────────────────────────────────────────────────────
    # Rate Features (ratios)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _add_rate_features(df: pd.DataFrame) -> pd.DataFrame:
        # Click concentration: IP-app clicks / total IP clicks
        if "clicks_ip" in df.columns and "clicks_ip_app" in df.columns:
            df["click_concentration"] = (
                df["clicks_ip_app"] / df["clicks_ip"].clip(lower=1)
            ).astype(np.float32)

        # Channel diversity ratio
        if "unique_channel_by_ip" in df.columns and "clicks_ip" in df.columns:
            df["channel_diversity_ratio"] = (
                df["unique_channel_by_ip"] / df["clicks_ip"].clip(lower=1)
            ).astype(np.float32)

        # Device diversity ratio
        if "unique_device_by_ip" in df.columns and "clicks_ip" in df.columns:
            df["device_diversity_ratio"] = (
                df["unique_device_by_ip"] / df["clicks_ip"].clip(lower=1)
            ).astype(np.float32)

        return df

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _get_feature_columns(df: pd.DataFrame) -> list:
        """Return columns suitable for model input (exclude meta/target)."""
        exclude = {
            "click_time", "attributed_time", "is_attributed",
            "click_id", "ip_app", "ip_device_os",
        }
        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in (
                np.float32, np.float64,
                np.int8, np.int16, np.int32, np.int64,
                np.uint8, np.uint16, np.uint32, np.uint64,
            )
        ]
        return sorted(feature_cols)
