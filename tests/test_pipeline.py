"""
Unit and Integration Tests for the Fraud Detection Engine.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataGenerator:
    """Tests for synthetic data generation."""

    def test_generates_correct_shape(self):
        from src.data_generator import generate_talkingdata_sample
        df = generate_talkingdata_sample(n_rows=1000, fraud_rate=0.01, save=False)
        assert len(df) == 1000

    def test_has_required_columns(self):
        from src.data_generator import generate_talkingdata_sample
        df = generate_talkingdata_sample(n_rows=1000, save=False)
        required = {"ip", "app", "device", "os", "channel", "click_time", "is_attributed"}
        assert required.issubset(set(df.columns))

    def test_fraud_rate_approximate(self):
        from src.data_generator import generate_talkingdata_sample
        df = generate_talkingdata_sample(n_rows=100_000, fraud_rate=0.01, save=False)
        actual_rate = df["is_attributed"].mean()
        assert 0.005 < actual_rate < 0.02  # within reasonable bounds

    def test_chronological_ordering(self):
        from src.data_generator import generate_talkingdata_sample
        df = generate_talkingdata_sample(n_rows=1000, save=False)
        times = pd.to_datetime(df["click_time"])
        assert times.is_monotonic_increasing


class TestFeatureEngineering:
    """Tests for the feature engineering pipeline."""

    @pytest.fixture
    def sample_df(self):
        from src.data_generator import generate_talkingdata_sample
        return generate_talkingdata_sample(n_rows=5000, save=False)

    def test_fit_transform_adds_features(self, sample_df):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert len(result.columns) > len(sample_df.columns)

    def test_feature_names_populated(self, sample_df):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        fe.fit_transform(sample_df)
        assert len(fe.feature_names) > 10

    def test_no_nulls_in_features(self, sample_df):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_df)
        feature_cols = fe.feature_names
        null_counts = result[feature_cols].isnull().sum()
        # Allow time delta features to have NaN (they use -1 fill)
        assert null_counts.sum() == 0

    def test_save_and_load(self, sample_df, tmp_path):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        fe.fit_transform(sample_df)

        save_path = str(tmp_path / "test_stats.joblib")
        fe.save(save_path)

        fe2 = FeatureEngineer()
        fe2.load(save_path)
        assert fe2.feature_names == fe.feature_names
        assert fe2.fitted is True


class TestModelTraining:
    """Tests for model training pipeline."""

    @pytest.fixture
    def training_data(self):
        from src.data_generator import generate_talkingdata_sample
        from src.feature_engineering import FeatureEngineer
        df = generate_talkingdata_sample(n_rows=10000, fraud_rate=0.02, save=False)
        fe = FeatureEngineer()
        df = fe.fit_transform(df)
        return df, fe.feature_names

    def test_chronological_split(self, training_data):
        from src.model_training import FraudModelTrainer
        df, feature_cols = training_data
        trainer = FraudModelTrainer()
        X_train, X_val, y_train, y_val = trainer.chronological_split(df, feature_cols)
        assert len(X_train) > len(X_val)
        assert len(y_train) == len(X_train)

    def test_lightgbm_trains(self, training_data):
        from src.model_training import FraudModelTrainer
        df, feature_cols = training_data
        trainer = FraudModelTrainer()
        X_train, X_val, y_train, y_val = trainer.chronological_split(df, feature_cols)
        model = trainer.train_lightgbm(X_train, X_val, y_train, y_val, feature_cols)
        assert model is not None
        assert "lgbm" in trainer.training_metrics

    def test_xgboost_trains(self, training_data):
        from src.model_training import FraudModelTrainer
        df, feature_cols = training_data
        trainer = FraudModelTrainer()
        X_train, X_val, y_train, y_val = trainer.chronological_split(df, feature_cols)
        model = trainer.train_xgboost(X_train, X_val, y_train, y_val, feature_cols)
        assert model is not None
        assert "xgb" in trainer.training_metrics


class TestRealTimeDetector:
    """Tests for the real-time fraud detection engine."""

    def test_predict_click_output_format(self):
        """Test that single-click prediction returns expected format."""
        # This test requires trained models — skip if not available
        from config.settings import LGBM_MODEL_PATH
        if not os.path.exists(LGBM_MODEL_PATH):
            pytest.skip("Models not trained yet")

        from src.fraud_detector import RealTimeFraudDetector
        detector = RealTimeFraudDetector()
        detector.load()

        click = {
            "ip": 87540,
            "app": 12,
            "device": 1,
            "os": 13,
            "channel": 497,
            "click_time": "2017-11-07T09:30:38",
        }
        result = detector.predict_click(click)

        assert "fraud_probability" in result
        assert "is_fraud" in result
        assert "risk_level" in result
        assert 0 <= result["fraud_probability"] <= 1
        assert result["risk_level"] in ["SAFE", "LOW", "MEDIUM", "HIGH"]

    def test_realtime_features(self):
        """Test sliding window feature computation."""
        from src.fraud_detector import RealTimeFraudDetector
        detector = RealTimeFraudDetector(window_seconds=60)

        click1 = {
            "ip": 100,
            "app": 5,
            "device": 1,
            "os": 10,
            "channel": 200,
            "click_time": "2017-11-07T09:30:38",
        }
        features = detector._update_realtime_state(click1)
        assert features["rt_ip_click_velocity"] == 1

        # Second click from same IP
        click2 = {**click1, "click_time": "2017-11-07T09:30:39"}
        features2 = detector._update_realtime_state(click2)
        assert features2["rt_ip_click_velocity"] == 2
        assert features2["rt_burst_count_1s"] >= 1

    def test_engine_stats(self):
        """Test engine statistics tracking."""
        from src.fraud_detector import RealTimeFraudDetector
        detector = RealTimeFraudDetector()
        stats = detector.get_engine_stats()
        assert stats["total_predictions"] == 0
        assert "avg_latency_ms" in stats


class TestFastAPIEndpoints:
    """Integration tests for the FastAPI REST interface."""

    @pytest.fixture(scope="module")
    def client(self):
        from fastapi.testclient import TestClient
        from src.api import app
        with TestClient(app) as test_client:
            yield test_client

    def test_health_endpoint(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "engine_stats" in data

    def test_predict_single_endpoint(self, client):
        payload = {
            "ip": 150000,
            "app": 3,
            "device": 1,
            "os": 19,
            "channel": 21,
            "click_time": "2017-11-07T12:00:00",
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "risk_level" in data
        assert data["risk_level"] in ["SAFE", "LOW", "MEDIUM", "HIGH"]
        assert data["latency_ms"] > 0

    def test_predict_batch_endpoint(self, client):
        payload = {
            "clicks": [
                {"ip": 150000, "app": 3, "device": 1, "os": 19, "channel": 21},
                {"ip": 698, "app": 12, "device": 1, "os": 13, "channel": 109},
            ]
        }
        res = client.post("/predict/batch", json=payload)
        if res.status_code != 200:
            print("BATCH ERROR DETAIL:", res.text)
        assert res.status_code == 200
        data = res.json()
        assert data["total_clicks"] == 2
        assert len(data["predictions"]) == 2
        assert "avg_latency_ms" in data

    def test_metrics_endpoint(self, client):
        res = client.get("/metrics")
        assert res.status_code == 200
        data = res.json()
        assert "ensemble" in data or "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
