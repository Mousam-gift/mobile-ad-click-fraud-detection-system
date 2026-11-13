"""
FastAPI Real-Time Prediction API.

Exposes the fraud detection engine as a REST API with:
  - Single click prediction
  - Batch prediction
  - Health check / stats endpoint
  - Dashboard endpoint
"""

import os
import sys
import time
import json
import uvicorn
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    API_HOST,
    API_PORT,
    FRAUD_THRESHOLD,
    MAX_BATCH_SIZE,
    TRAINING_METRICS_PATH,
    PROJECT_ROOT,
)
from src.fraud_detector import RealTimeFraudDetector

detector = RealTimeFraudDetector()
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for model loading on startup."""
    try:
        detector.load()
        print("[API] Fraud detection engine loaded and ready")
    except Exception as e:
        print(f"[API] Warning: Could not load models: {e}")
        print("[API] Run train_pipeline.py first to train models")
    yield


# ─────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="🛡️ Ad Click Fraud Detection API",
    description=(
        "Real-time binary classification engine for mobile ad click fraud detection. "
        "Uses a LightGBM + XGBoost ensemble trained on TalkingData patterns."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

if os.path.exists(PLOT_DIR):
    app.mount("/plots", StaticFiles(directory=PLOT_DIR), name="plots")


# ─────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────
class ClickEvent(BaseModel):
    ip: int = Field(..., ge=0, description="IP address (anonymized integer)")
    app: int = Field(..., ge=0, description="Application ID")
    device: int = Field(..., ge=0, description="Device type ID")
    os: int = Field(..., ge=0, description="OS version ID")
    channel: int = Field(..., ge=0, description="Ad channel ID")
    click_time: str = Field(
        default="2017-11-07T09:30:00",
        description="Click timestamp (ISO 8601)",
    )
    track_realtime: bool = Field(
        default=False,
        description="Whether to track in sliding window (True for live streams, False for isolated single-click audit)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ip": 83230,
                    "app": 3,
                    "device": 1,
                    "os": 19,
                    "channel": 379,
                    "click_time": "2017-11-07T09:30:00",
                    "track_realtime": False,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    fraud_probability: float
    lgbm_probability: float
    xgb_probability: float
    is_fraud: bool
    risk_level: str
    latency_ms: float
    real_time_features: dict
    click: dict


class BatchRequest(BaseModel):
    clicks: List[ClickEvent] = Field(..., max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_clicks: int
    fraud_count: int
    avg_latency_ms: float


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Interactive Real-Time Fraud Detection Dashboard."""
    template_path = os.path.join(TEMPLATE_DIR, "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ShieldGuard AI Dashboard</h1><p>Template not found.</p>"


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(click: ClickEvent):
    """
    Predict fraud probability for a single click event.

    Returns fraud probability, binary classification, and risk level.
    """
    try:
        data = click.model_dump()
        track_rt = data.pop("track_realtime", False)
        result = detector.predict_click(data, track_realtime=track_rt)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest):
    """
    Predict fraud for a batch of click events (up to 1000).
    """
    try:
        clicks_data = []
        for c in batch.clicks:
            d = c.model_dump()
            tr = d.pop("track_realtime", True)
            clicks_data.append((d, tr))

        results = [
            detector.predict_click(d, track_realtime=tr)
            for d, tr in clicks_data
        ]
        fraud_count = sum(1 for r in results if r["is_fraud"])
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)

        return BatchResponse(
            predictions=[PredictionResponse(**r) for r in results],
            total_clicks=len(results),
            fraud_count=fraud_count,
            avg_latency_ms=round(avg_latency, 3),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check with engine statistics."""
    stats = detector.get_engine_stats()
    return {
        "status": "healthy" if detector._loaded else "models_not_loaded",
        "model_loaded": detector._loaded,
        "fraud_threshold": FRAUD_THRESHOLD,
        "engine_stats": stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics")
async def get_metrics():
    """Return training performance metrics."""
    if os.path.exists(TRAINING_METRICS_PATH):
        with open(TRAINING_METRICS_PATH) as f:
            return json.load(f)
    return {"error": "No training metrics found. Train the model first."}


@app.get("/simulate")
async def simulate_traffic():
    """Simulate a burst of clicks for demonstration."""
    import random
    clicks = []
    for i in range(20):
        click = {
            "ip": random.choice([100, 200, 300, 87540, 55555]),
            "app": random.choice([12, 15, 18, 3, 9, 2, 35]),
            "device": random.choice([1, 2, 3]),
            "os": random.choice([13, 19, 25, 32]),
            "channel": random.choice([107, 128, 245, 477, 497]),
            "click_time": datetime.now(timezone.utc).isoformat(),
        }
        clicks.append(click)

    results = detector.predict_batch(clicks)
    fraud_count = sum(1 for r in results if r["is_fraud"])

    return {
        "simulated_clicks": len(clicks),
        "fraud_detected": fraud_count,
        "fraud_rate": f"{fraud_count/len(clicks)*100:.1f}%",
        "predictions": results,
    }


# ─────────────────────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────────────────────
def start_api():
    """Start the API server."""
    port = int(os.environ.get("PORT", API_PORT))
    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=port,
        reload=False,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    start_api()
