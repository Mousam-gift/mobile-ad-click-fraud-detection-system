"""
Interactive Demo Script.

Demonstrates the full fraud detection pipeline:
  1. Trains models (if not already trained)
  2. Runs real-time predictions on simulated traffic
  3. Shows detection of various fraud patterns
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import LGBM_MODEL_PATH, FRAUD_THRESHOLD
from src.fraud_detector import RealTimeFraudDetector


def print_header(text: str):
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)


def print_result(result: dict, idx: int = 0):
    """Pretty-print a single prediction result."""
    click = result["click"]
    risk_emoji = {
        "SAFE": "✅", "LOW": "⚠️ ", "MEDIUM": "🟠", "HIGH": "🔴"
    }
    emoji = risk_emoji.get(result["risk_level"], "❓")

    print(f"\n  Click #{idx+1}:")
    print(f"    IP: {click['ip']:>8} | App: {click['app']:>4} | "
          f"Device: {click['device']:>4} | OS: {click['os']:>4} | "
          f"Channel: {click['channel']:>4}")
    print(f"    Fraud Prob: {result['fraud_probability']:.4f} "
          f"({result['lgbm_probability']:.4f} LGB / {result['xgb_probability']:.4f} XGB)")
    print(f"    {emoji} Risk: {result['risk_level']} | "
          f"Fraud: {'YES' if result['is_fraud'] else 'no'} | "
          f"Latency: {result['latency_ms']:.2f}ms")


def demo_normal_traffic(detector: RealTimeFraudDetector):
    """Simulate normal user click behavior."""
    print_header("🟢  SCENARIO 1: Normal User Traffic")
    print("  Simulating organic user clicks from diverse IPs/devices...\n")

    rng = random.Random(42)
    base_time = datetime(2017, 11, 7, 14, 30, 0)

    clicks = []
    for i in range(10):
        clicks.append({
            "ip": rng.randint(100000, 280000),
            "app": rng.choice([3, 9, 12, 15, 18, 25, 50, 100, 200, 350]),
            "device": rng.choice([1, 2, 3, 4, 5, 10, 50, 100]),
            "os": rng.choice([13, 17, 19, 22, 25, 29, 32, 38]),
            "channel": rng.choice([21, 107, 128, 245, 259, 343, 442, 477]),
            "click_time": (base_time + timedelta(minutes=i * rng.randint(5, 60))).isoformat(),
        })

    results = detector.predict_batch(clicks)
    for i, r in enumerate(results):
        print_result(r, i)

    fraud_count = sum(1 for r in results if r["is_fraud"])
    print(f"\n  📊 Summary: {fraud_count}/{len(results)} flagged as fraud")


def demo_bot_burst(detector: RealTimeFraudDetector):
    """Simulate a bot farm rapid-clicking pattern."""
    print_header("🔴  SCENARIO 2: Bot Farm — Rapid Click Burst")
    print("  Same IP generating 15 clicks in 3 seconds on same app...\n")

    bot_ip = 555
    base_time = datetime(2017, 11, 7, 3, 15, 0)

    clicks = []
    for i in range(15):
        clicks.append({
            "ip": bot_ip,
            "app": 12,
            "device": 1,
            "os": 19,
            "channel": 128,
            "click_time": (base_time + timedelta(milliseconds=i * 200)).isoformat(),
        })

    results = detector.predict_batch(clicks)
    for i, r in enumerate(results):
        print_result(r, i)

    fraud_count = sum(1 for r in results if r["is_fraud"])
    print(f"\n  📊 Summary: {fraud_count}/{len(results)} flagged as fraud")


def demo_click_farm(detector: RealTimeFraudDetector):
    """Simulate a click farm: few IPs, same apps, low device diversity."""
    print_header("🟠  SCENARIO 3: Click Farm — Coordinated Fraud")
    print("  Known bot farm IPs clicking on high-value apps with identical devices...\n")

    farm_ips = [698, 81, 833]
    base_time = datetime(2017, 11, 7, 2, 0, 0)

    clicks = []
    for i in range(12):
        clicks.append({
            "ip": farm_ips[i % 3],
            "app": random.choice([12, 15, 18]),
            "device": 1,  # all same device
            "os": 13,
            "channel": random.choice([107, 128, 109]),
            "click_time": (base_time + timedelta(seconds=i * 2)).isoformat(),
        })

    results = detector.predict_batch(clicks)
    for i, r in enumerate(results):
        print_result(r, i)

    fraud_count = sum(1 for r in results if r["is_fraud"])
    print(f"\n  📊 Summary: {fraud_count}/{len(results)} flagged as fraud")


def demo_mixed_traffic(detector: RealTimeFraudDetector):
    """Simulate mixed legitimate + fraudulent traffic."""
    print_header("🔄  SCENARIO 4: Mixed Traffic Stream")
    print("  Real-time stream with interspersed legitimate and fraud clicks...\n")

    base_time = datetime(2017, 11, 7, 10, 0, 0)
    clicks = []
    labels = []

    # Legitimate clicks
    for i in range(8):
        clicks.append({
            "ip": random.randint(50000, 200000),
            "app": random.choice([3, 9, 50, 100, 200]),
            "device": random.choice([1, 2, 3, 5, 10, 50]),
            "os": random.choice([13, 19, 25, 32]),
            "channel": random.choice([21, 245, 343, 477]),
            "click_time": (base_time + timedelta(minutes=i * 10)).isoformat(),
        })
        labels.append("LEGIT")

    # Inject fraud clicks (from bot farm IP 698)
    for i in range(5):
        clicks.append({
            "ip": 698,
            "app": 12,
            "device": 1,
            "os": 13,
            "channel": 109,
            "click_time": (base_time + timedelta(seconds=50 + i)).isoformat(),
        })
        labels.append("FRAUD")

    # Shuffle
    combined = list(zip(clicks, labels))
    random.shuffle(combined)
    clicks, labels = zip(*combined)

    results = detector.predict_batch(list(clicks))
    for i, (r, label) in enumerate(zip(results, labels)):
        print_result(r, i)
        print(f"    📋 Actual: {label}")

    fraud_count = sum(1 for r in results if r["is_fraud"])
    print(f"\n  📊 Summary: {fraud_count}/{len(results)} flagged as fraud")


def demo_engine_stats(detector: RealTimeFraudDetector):
    """Display engine performance statistics."""
    print_header("📈  ENGINE PERFORMANCE STATISTICS")

    stats = detector.get_engine_stats()
    print(f"""
    Total Predictions:     {stats['total_predictions']:,}
    Total Fraud Flagged:   {stats['total_fraud_flagged']:,}
    Detection Rate:        {stats['fraud_rate']:.2%}
    Tracked IPs:           {stats['tracked_ips']:,}

    Latency:
      Average:             {stats['avg_latency_ms']:.3f} ms
      P50:                 {stats['p50_latency_ms']:.3f} ms
      P95:                 {stats['p95_latency_ms']:.3f} ms
      P99:                 {stats['p99_latency_ms']:.3f} ms
    """)


def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█   🛡️  MOBILE AD CLICK FRAUD DETECTION ENGINE — LIVE DEMO       █")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Check if models exist
    if not os.path.exists(LGBM_MODEL_PATH):
        print("\n⚠️  No trained models found. Running training pipeline first...\n")
        from train_pipeline import main as train_main
        train_main()
        print("\n✅  Training complete! Starting demo...\n")

    # Load detector
    detector = RealTimeFraudDetector(window_seconds=300)
    detector.load()

    # Run all scenarios
    demo_normal_traffic(detector)
    time.sleep(0.5)

    demo_bot_burst(detector)
    time.sleep(0.5)

    demo_click_farm(detector)
    time.sleep(0.5)

    demo_mixed_traffic(detector)

    demo_engine_stats(detector)

    print("\n" + "═" * 70)
    print("  ✅  DEMO COMPLETE")
    print("  Start the API server with: python -m src.api")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
