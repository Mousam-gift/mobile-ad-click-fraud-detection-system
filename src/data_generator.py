"""
Synthetic Data Generator for TalkingData Ad Click Fraud Detection.

Generates realistic synthetic data matching the TalkingData schema for
development, testing, and demonstration when the Kaggle dataset is not
available locally. Mimics real-world fraud patterns and class imbalance.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_DIR, DTYPES


def generate_talkingdata_sample(
    n_rows: int = 500_000,
    fraud_rate: float = 0.0025,   # ~0.25% positive rate (realistic)
    seed: int = 42,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset matching TalkingData's schema with
    realistic fraud patterns baked in.

    Fraud patterns injected:
      1. IP-burst: Certain IPs produce abnormally high click volumes
      2. Device-channel combos: Specific device+channel combos are bot-heavy
      3. Temporal bursts: Fraud clicks cluster in tight time windows
      4. Low device diversity: Fraud IPs use fewer unique devices

    Parameters
    ----------
    n_rows : int
        Total number of rows to generate.
    fraud_rate : float
        Base fraud (is_attributed=1) rate.
    seed : int
        Random seed for reproducibility.
    save : bool
        If True, write CSV to DATA_DIR.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    # ── Legitimate clicks ──────────────────────────────────────────
    legit_ips = rng.integers(1, 280_000, size=n_legit).astype(np.uint32)
    legit_apps = rng.choice(np.arange(1, 800), size=n_legit, p=_zipf_probs(800 - 1, rng)).astype(np.uint16)
    legit_devices = rng.choice(np.arange(1, 4000), size=n_legit, p=_zipf_probs(4000 - 1, rng)).astype(np.uint16)
    legit_os = rng.choice(np.arange(1, 900), size=n_legit, p=_zipf_probs(900 - 1, rng)).astype(np.uint16)
    legit_channels = rng.choice(np.arange(1, 500), size=n_legit, p=_zipf_probs(500 - 1, rng)).astype(np.uint16)

    # Spread across 3 days
    base_time = datetime(2017, 11, 7, 0, 0, 0)
    legit_seconds = np.sort(rng.integers(0, 3 * 86400, size=n_legit))
    legit_times = pd.to_datetime(
        [base_time + timedelta(seconds=int(s)) for s in legit_seconds]
    )

    # ── Fraudulent clicks ──────────────────────────────────────────
    # Pattern 1: Small pool of IPs (bot farms)
    fraud_ip_pool = rng.integers(1, 1000, size=50).astype(np.uint32)
    fraud_ips = rng.choice(fraud_ip_pool, size=n_fraud).astype(np.uint32)

    # Pattern 2: Concentrate on high-value apps
    top_apps = np.array([12, 15, 18, 2, 3, 9, 35], dtype=np.uint16)
    fraud_apps = rng.choice(top_apps, size=n_fraud).astype(np.uint16)

    # Pattern 3: Low device diversity
    fraud_device_pool = np.array([1, 2], dtype=np.uint16)
    fraud_devices = rng.choice(fraud_device_pool, size=n_fraud).astype(np.uint16)

    fraud_os = rng.choice(np.arange(1, 50), size=n_fraud).astype(np.uint16)
    fraud_channels = rng.choice(np.arange(100, 200), size=n_fraud).astype(np.uint16)

    # Pattern 4: Temporal bursts (clusters within seconds)
    n_bursts = max(1, n_fraud // 10)
    burst_starts = rng.integers(0, 3 * 86400, size=n_bursts)
    burst_parts = [
        rng.integers(s, s + 10, size=min(10, n_fraud))
        for s in burst_starts
    ]
    burst_seconds = np.concatenate(burst_parts) if burst_parts else np.array([], dtype=np.int64)
    fraud_seconds = np.sort(burst_seconds[:n_fraud])
    if len(fraud_seconds) < n_fraud:
        extra = rng.integers(0, 3 * 86400, size=n_fraud - len(fraud_seconds))
        fraud_seconds = np.sort(np.concatenate([fraud_seconds, extra]))

    fraud_times = pd.to_datetime(
        [base_time + timedelta(seconds=int(s)) for s in fraud_seconds]
    )

    # ── Assemble DataFrame ─────────────────────────────────────────
    df = pd.DataFrame({
        "ip": np.concatenate([legit_ips, fraud_ips]),
        "app": np.concatenate([legit_apps, fraud_apps]),
        "device": np.concatenate([legit_devices, fraud_devices]),
        "os": np.concatenate([legit_os, fraud_os]),
        "channel": np.concatenate([legit_channels, fraud_channels]),
        "click_time": pd.concat([
            pd.Series(legit_times), pd.Series(fraud_times)
        ], ignore_index=True),
        "is_attributed": np.concatenate([
            np.zeros(n_legit, dtype=np.uint8),
            np.ones(n_fraud, dtype=np.uint8),
        ]),
    })

    # Add attributed_time only for is_attributed=1
    df["attributed_time"] = pd.NaT
    mask = df["is_attributed"] == 1
    df.loc[mask, "attributed_time"] = df.loc[mask, "click_time"] + pd.to_timedelta(
        rng.integers(10, 600, size=mask.sum()), unit="s"
    )

    # Sort chronologically (mimics real stream)
    df = df.sort_values("click_time").reset_index(drop=True)

    # Cast types
    for col, dtype in DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, "train_sample.csv")
        df.to_csv(path, index=False)
        print(f"[DataGen] Saved {len(df):,} rows → {path}")
        print(f"[DataGen] Fraud rate: {df['is_attributed'].mean():.4%}")

    return df


def _zipf_probs(n: int, rng) -> np.ndarray:
    """Generate Zipf-like probability distribution for categorical features."""
    ranks = np.arange(1, n + 1, dtype=np.float64)
    probs = 1.0 / (ranks ** 1.07)
    probs /= probs.sum()
    return probs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic TalkingData sample")
    parser.add_argument("--rows", type=int, default=500_000, help="Number of rows")
    parser.add_argument("--fraud-rate", type=float, default=0.0025, help="Fraud rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = generate_talkingdata_sample(
        n_rows=args.rows,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )
    print(f"\nClass distribution:\n{df['is_attributed'].value_counts()}")
    print(f"\nSample:\n{df.head()}")
