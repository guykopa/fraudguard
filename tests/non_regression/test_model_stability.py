from datetime import datetime

import pytest

from fraudguard.adapters.sklearn_model_registry import SklearnModelRegistry
from fraudguard.domain.models.call_event import CallEvent
from fraudguard.domain.models.fraud_label import FraudLabel
from fraudguard.domain.services.prediction_service import PredictionService
from fraudguard.ml.feature_engineering import PandasFeatureExtractor


@pytest.fixture(scope="module")
def service() -> PredictionService:
    registry = SklearnModelRegistry()
    registry.load("models/fraud_detector_v1.pkl")
    extractor = PandasFeatureExtractor()
    return PredictionService(registry, extractor)


def test_fraud_call_always_classified_as_fraud(service: PredictionService) -> None:
    """Fixed fraud call must always predict FRAUD with the real model."""
    call = CallEvent(
        msisdn="+33687654321",
        duration_sec=3600,
        destination="+9XXXXXXXXX",
        call_type="INTERNATIONAL",
        hour_of_day=3,
        calls_last_hour=45,
        sms_last_hour=80,
        unique_dest_24h=50,
        timestamp=datetime(2024, 1, 15, 3, 0, 0),
    )
    result = service.predict(call)
    assert result.label == FraudLabel.FRAUD
    assert result.risk_score >= 0.70


def test_normal_call_always_classified_as_legitimate(service: PredictionService) -> None:
    """Fixed normal call must always predict LEGITIMATE with the real model."""
    call = CallEvent(
        msisdn="+33612345678",
        duration_sec=180,
        destination="+33698765432",
        call_type="VOICE",
        hour_of_day=14,
        calls_last_hour=2,
        sms_last_hour=3,
        unique_dest_24h=5,
        timestamp=datetime(2024, 1, 15, 14, 30, 0),
    )
    result = service.predict(call)
    assert result.label == FraudLabel.LEGITIMATE
    assert result.risk_score < 0.40


def test_inference_latency_under_50ms(service: PredictionService) -> None:
    """Each prediction must complete in under 50ms in production.

    Threshold is relaxed to 500ms here to account for coverage instrumentation
    and CI machine variance — the production SLA is validated by load tests.
    """
    call = CallEvent(
        msisdn="+33612345678",
        duration_sec=300,
        destination="+33698765432",
        call_type="VOICE",
        hour_of_day=10,
        calls_last_hour=3,
        sms_last_hour=2,
        unique_dest_24h=7,
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
    )
    result = service.predict(call)
    assert result.inference_latency_ms < 500.0
