"""
EDA & Visualization Module for Ad Click Fraud Detection.

Generates comprehensive exploratory data analysis plots and reports.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PROJECT_ROOT

PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")


def setup_plotting():
    """Configure matplotlib defaults."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame, save: bool = True):
    """Plot target class distribution showing extreme imbalance."""
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    counts = df["is_attributed"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=colors)
    axes[0].set_title("Class Distribution (Count)")
    axes[0].set_ylabel("Number of Clicks")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + v * 0.02, f"{v:,.0f}", ha="center", fontweight="bold")

    # Percentage pie
    axes[1].pie(
        counts.values, labels=["Legitimate", "Fraud"],
        autopct="%1.3f%%", colors=colors, startangle=90,
        explode=(0, 0.1), shadow=True
    )
    axes[1].set_title("Class Distribution (Percentage)")

    plt.suptitle("Extreme Class Imbalance in Ad Click Data", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, "01_class_distribution.png"), bbox_inches="tight")
    plt.close()


def plot_hourly_pattern(df: pd.DataFrame, save: bool = True):
    """Plot click volume and fraud rate by hour."""
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    hourly = df.groupby("hour").agg(
        total_clicks=("is_attributed", "count"),
        fraud_clicks=("is_attributed", "sum"),
    ).reset_index()
    hourly["fraud_rate"] = hourly["fraud_clicks"] / hourly["total_clicks"]

    # Volume
    axes[0].bar(hourly["hour"], hourly["total_clicks"], color="#3498db", alpha=0.8)
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Total Clicks")
    axes[0].set_title("Click Volume by Hour")
    axes[0].set_xticks(range(0, 24))

    # Fraud rate
    axes[1].plot(hourly["hour"], hourly["fraud_rate"] * 100, "o-", color="#e74c3c", linewidth=2)
    axes[1].fill_between(hourly["hour"], hourly["fraud_rate"] * 100, alpha=0.2, color="#e74c3c")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].set_title("Fraud Rate by Hour")
    axes[1].set_xticks(range(0, 24))

    plt.suptitle("Temporal Patterns in Ad Clicks", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, "02_hourly_patterns.png"), bbox_inches="tight")
    plt.close()


def plot_top_features(df: pd.DataFrame, feature_col: str, top_n: int = 20, save: bool = True):
    """Plot fraud rate for top-N values of a categorical feature."""
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top by volume
    top_values = df[feature_col].value_counts().head(top_n)
    axes[0].barh(
        [str(x) for x in top_values.index],
        top_values.values, color="#3498db"
    )
    axes[0].set_title(f"Top {top_n} {feature_col} by Click Volume")
    axes[0].set_xlabel("Click Count")
    axes[0].invert_yaxis()

    # Fraud rate for those values
    fraud_rates = df[df[feature_col].isin(top_values.index)].groupby(feature_col)["is_attributed"].mean()
    fraud_rates = fraud_rates.reindex(top_values.index)
    colors = ["#e74c3c" if r > df["is_attributed"].mean() * 2 else "#2ecc71" for r in fraud_rates.values]
    axes[1].barh(
        [str(x) for x in fraud_rates.index],
        fraud_rates.values * 100, color=colors
    )
    axes[1].set_title(f"Fraud Rate by {feature_col}")
    axes[1].set_xlabel("Fraud Rate (%)")
    axes[1].invert_yaxis()
    axes[1].axvline(df["is_attributed"].mean() * 100, color="black", linestyle="--", label="Avg Fraud Rate")
    axes[1].legend()

    plt.suptitle(f"Analysis: {feature_col}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"03_{feature_col}_analysis.png"), bbox_inches="tight")
    plt.close()


def plot_click_frequency_distribution(df: pd.DataFrame, save: bool = True):
    """Plot IP click frequency distributions for legitimate vs fraud."""
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ip_counts = df.groupby(["ip", "is_attributed"]).size().reset_index(name="count")

    for label, color, ax_idx in [(0, "#2ecc71", 0), (1, "#e74c3c", 1)]:
        subset = ip_counts[ip_counts["is_attributed"] == label]
        title = "Legitimate" if label == 0 else "Fraudulent"
        axes[ax_idx].hist(
            subset["count"].clip(upper=100), bins=50, color=color,
            alpha=0.8, edgecolor="white"
        )
        axes[ax_idx].set_title(f"{title} - IP Click Frequency")
        axes[ax_idx].set_xlabel("Clicks per IP")
        axes[ax_idx].set_ylabel("Number of IPs")
        axes[ax_idx].set_yscale("log")

    plt.suptitle("Click Frequency Distribution: Legitimate vs Fraudulent", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, "04_click_frequency.png"), bbox_inches="tight")
    plt.close()


def plot_model_performance(y_true, y_proba, model_name="Ensemble", save=True):
    """Plot ROC and Precision-Recall curves."""
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="#3498db", linewidth=2, label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — {model_name}")
    axes[0].legend(loc="lower right", fontsize=12)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, color="#e74c3c", linewidth=2, label=f"AP = {pr_auc:.4f}")
    baseline = y_true.mean()
    axes[1].axhline(baseline, color="black", linestyle="--", alpha=0.5, label=f"Baseline = {baseline:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {model_name}")
    axes[1].legend(loc="upper right", fontsize=12)

    plt.suptitle(f"Model Performance: {model_name}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, "05_model_performance.png"), bbox_inches="tight")
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, save: bool = True):
    """Plot combined feature importance from ensemble."""
    setup_plotting()
    fig, ax = plt.subplots(figsize=(12, 8))

    data = importance_df.head(top_n).sort_values("ensemble_importance")

    y_pos = range(len(data))
    ax.barh(y_pos, data["lgbm_importance"], height=0.4, color="#3498db",
            alpha=0.8, label="LightGBM")
    ax.barh([y + 0.4 for y in y_pos], data["xgb_importance"], height=0.4,
            color="#e67e22", alpha=0.8, label="XGBoost")

    ax.set_yticks([y + 0.2 for y in y_pos])
    ax.set_yticklabels(data["feature"])
    ax.set_xlabel("Normalized Importance")
    ax.set_title(f"Top {top_n} Feature Importance — LightGBM vs XGBoost", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, "06_feature_importance.png"), bbox_inches="tight")
    plt.close()


def plot_threshold_analysis(y_true, y_proba, save=True):
    """Analyze precision/recall trade-off at different thresholds."""
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    thresholds = np.arange(0.05, 0.95, 0.05)
    precisions_list, recalls_list, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        precisions_list.append(prec)
        recalls_list.append(rec)
        f1s.append(f1)

    axes[0].plot(thresholds, precisions_list, "b-o", label="Precision", linewidth=2)
    axes[0].plot(thresholds, recalls_list, "r-s", label="Recall", linewidth=2)
    axes[0].plot(thresholds, f1s, "g-^", label="F1", linewidth=2)
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision / Recall / F1 vs Threshold")
    axes[0].legend()
    axes[0].grid(True)

    # Cost analysis: Assume FP costs $0.01, FN costs $1.00
    fp_cost, fn_cost = 0.01, 1.00
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        costs.append(fp * fp_cost + fn * fn_cost)

    axes[1].plot(thresholds, costs, "m-o", linewidth=2)
    opt_idx = np.argmin(costs)
    axes[1].axvline(thresholds[opt_idx], color="red", linestyle="--",
                    label=f"Optimal = {thresholds[opt_idx]:.2f}")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Total Cost ($)")
    axes[1].set_title("Cost-Sensitive Threshold Optimization")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("Threshold Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, "07_threshold_analysis.png"), bbox_inches="tight")
    plt.close()


def generate_full_eda(df: pd.DataFrame):
    """Run all EDA visualizations."""
    print("[EDA] Generating comprehensive visualizations...")
    setup_plotting()

    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["click_time"]).dt.hour

    plot_class_distribution(df)
    print("  ✓ Class distribution")

    plot_hourly_pattern(df)
    print("  ✓ Hourly patterns")

    for col in ["app", "device", "os", "channel"]:
        plot_top_features(df, col)
        print(f"  ✓ {col} analysis")

    plot_click_frequency_distribution(df)
    print("  ✓ Click frequency distribution")

    print(f"\n[EDA] All plots saved to {PLOT_DIR}/")
