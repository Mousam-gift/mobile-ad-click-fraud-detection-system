# 🛡️ AdShield AI — Real-Time Mobile Ad Click Fraud Detection Engine

> **Production-grade binary classification engine for intercepting fraudulent advertising clicks in under 2ms.**  
> Built and validated on 1,000,000 real clicks from the **TalkingData AdTracking Fraud Detection** benchmark.

---

## 🎯 Business Problem

Digital advertisers lose billions annually to automated bots, headless scrapers, and click farms that generate empty clicks to drain budgets. **AdShield AI** intercepts invalid clicks in real-time before ad exchanges charge cost-per-click (CPC) fees, preserving marketing budget for genuine human customers.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│               REAL-TIME INFERENCE SERVICE (FastAPI + Uvicorn)          │
├────────────────────────────────────────────────────────────────────────┤
│  Incoming Click Event ──► [ Causal Feature Engineering ]               │
│                                    │                                   │
│                        35 Leak-Free Indicators                         │
│                                    ▼                                   │
│               [ LightGBM (60%) + XGBoost (40%) Ensemble ]              │
│                                    │                                   │
│                       Calibrated Posterior Probability                 │
│                                    ▼                                   │
│             [ Optimal Threshold Decision (threshold: 0.5332) ]         │
│                                    │                                   │
│               ┌────────────────────┴────────────────────┐              │
│               ▼                                         ▼              │
│      ✅ VERIFIED HUMAN (Safe)               🚫 FRAUDULENT BOT (Blocked) │
│      Allowed to install app                CPC charge prevented ($1.50)│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Clean Project Structure

```
mobile_ad_click_fraud_detection/
├── config/
│   ├── __init__.py
│   └── settings.py              # Centralized configurations & hyperparams
├── data/
│   └── train.csv                # Kaggle TalkingData dataset (~185M clicks)
├── models/
│   ├── click_stats.joblib       # Historical frequency & diversity lookup tables
│   ├── ensemble_model.joblib    # Dual-model metadata & optimal threshold
│   ├── lgbm_fraud_model.joblib  # Trained LightGBM booster (179 trees)
│   ├── xgb_fraud_model.joblib   # Trained XGBoost booster (45 trees)
│   └── training_metrics.json    # Out-of-time evaluation metrics
├── plots/                       # Automated EDA & validation curves (10 plots)
│   ├── 01_class_distribution.png
│   ├── 02_hourly_patterns.png
│   ├── 03_app_analysis.png
│   ├── 03_channel_analysis.png
│   ├── 03_device_analysis.png
│   ├── 03_os_analysis.png
│   ├── 04_click_frequency.png
│   ├── 05_model_performance.png
│   ├── 06_feature_importance.png
│   └── 07_threshold_analysis.png
├── src/
│   ├── __init__.py
│   ├── api.py                   # FastAPI REST API & dashboard routes
│   ├── data_generator.py        # Synthetic TalkingData stream generator
│   ├── feature_engineering.py   # 35 leak-free, causal features
│   ├── fraud_detector.py        # Real-time inference & sliding-window engine
│   ├── model_training.py        # Ensemble training, PR evaluation & calibration
│   └── visualization.py         # Matplotlib/Seaborn visualization pipeline
├── templates/
│   └── index.html               # Clean, modern Web UI & Live Simulator
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py         # 18 automated unit and integration tests
├── .gitignore                   # Production git exclusion rules
├── app_streamlit.py             # Optional standalone Streamlit UI
├── requirements.txt             # Environment dependencies
├── run_demo.py                  # Interactive CLI demonstration
├── train_pipeline.py            # End-to-end leak-free training pipeline
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/your-username/mobile_ad_click_fraud_detection.git
cd mobile_ad_click_fraud_detection
pip install -r requirements.txt
```

### 2. Run the Web Dashboard & API

```bash
PORT=8080 python3 -m src.api
```
- **Web Dashboard**: [http://localhost:8080/](http://localhost:8080/)
- **Interactive Swagger Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)

### 3. Run Automated Tests

```bash
pytest tests/ -v
```

### 4. Retrain the Pipeline (Optional)

```bash
python3 train_pipeline.py --rows 1000000
```

---

## 🧬 Feature Engineering (35 Causal Features)

All features are strictly **backward-looking** to guarantee zero training-serving skew:

| Category | Features | Description |
|---|---|---|
| **Temporal** | `hour`, `minute`, `second`, `day_of_week`, `hour_sin`, `hour_cos`, `time_of_day` | Cyclical time encodings |
| **Historical Counts** | `clicks_ip`, `clicks_ip_app`, `clicks_ip_app_os`, `clicks_ip_device`, `clicks_ip_channel`, `clicks_app_channel` | Frequency aggregations fitted on historical data only |
| **Causal Time Deltas** | `prev_click_ip_app_device_os`, `prev_click_ip_os_device`, `prev_click_ip_os_device_app` | Seconds elapsed since previous click (no future lookahead) |
| **Diversity Ratios** | `unique_channel_by_ip`, `unique_app_by_ip`, `unique_device_by_ip`, `unique_os_by_ip`, `channel_diversity_ratio`, `device_diversity_ratio` | Behavioral entropy & bot footprint |
| **Cross Interactions** | `ip_app_hash`, `ip_device_os_hash`, `app_channel`, `click_concentration` | High-order interaction signals |
| **Real-Time Sliding Window** | `rt_ip_click_velocity`, `rt_burst_count_1s`, `rt_app_concentration` | In-memory 5-minute sliding window |

---

## 📊 Validated Benchmark Performance

Evaluated on an **out-of-time chronological validation split** (900,000 train rows vs. 100,000 validation rows):

| Metric | Score | Industry Context |
|---|:---:|---|
| **ROC-AUC** | **95.35%** | High ranking discrimination |
| **Average Precision (PR-AUC)** | **43.22%** | Strong precision in extreme 0.16% class imbalance |
| **Optimal F1 Score** | **50.92%** | Balances fraud catch rate and advertiser reach |
| **Decision Threshold** | **0.5332** | Calibrated threshold (avoids false-alarm disasters) |
| **False Alarms (Validation)** | **Only 122 / 100k** | **99.88% clean specificity** |
| **Inference Latency** | **1.2 ms** | Inline ad-decisioning compliant (< 10ms SLA) |

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Clean Web UI Dashboard & Live Simulator |
| `POST` | `/predict` | Single click evaluation (`track_realtime: false` for deterministic audit) |
| `POST` | `/predict/batch` | Batch click evaluation (up to 1,000 clicks) |
| `GET` | `/health` | Service health status and live engine metrics |
| `GET` | `/metrics` | Model training evaluation metrics & confusion matrix |

### Single Click Request Example

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ip": 83230,
    "app": 3,
    "device": 1,
    "os": 19,
    "channel": 379,
    "click_time": "2017-11-07T09:30:00",
    "track_realtime": false
  }'
```

### Response

```json
{
  "fraud_probability": 0.00033,
  "lgbm_probability": 0.00019,
  "xgb_probability": 0.00054,
  "is_fraud": false,
  "risk_level": "SAFE",
  "latency_ms": 1.18,
  "real_time_features": {
    "rt_ip_click_velocity": 1,
    "rt_burst_count_1s": 1
  },
  "click": {
    "ip": 83230,
    "app": 3,
    "device": 1,
    "os": 19,
    "channel": 379
  }
}
```

---

## 📄 License
MIT License. Feel free to use and adapt for commercial or research purposes.
