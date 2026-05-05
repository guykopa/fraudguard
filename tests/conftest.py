import pytest
import numpy as np
from datetime import datetime

from fraudguard.domain.models.call_event import CallEvent
from fraudguard.domain.models.feature_vector import FeatureVector
from fraudguard.interfaces.i_model_registry import IModelRegistry
from fraudguard.interfaces.i_feature_extractor import IFeatureExtractor


# ---------------------------------------------------------------------------
# CallEvent fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_call() -> CallEvent:
    """Normal legitimate call — short duration, daytime, domestic."""
    return CallEvent(
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


@pytest.fixture
def fraud_call() -> CallEvent:
    """Suspicious fraud call — long duration, night, international, high volume."""
    return CallEvent(
        msisdn="+33687654321",
        duration_sec=3600,
        destination="+9XXXXXXXXX",
        call_type="INTERNATIONAL",
        hour_of_day=3,
        calls_last_hour=45,
        sms_last_hour=80,
        unique_dest_24h=50,
        timestamp=datetime(2024, 1, 15, 3, 15, 0),
    )


@pytest.fixture
def night_call() -> CallEvent:
    """Night call — hour_of_day in [0-5]."""
    return CallEvent(
        msisdn="+33612345678",
        duration_sec=300,
        destination="+33698765432",
        call_type="VOICE",
        hour_of_day=2,
        calls_last_hour=1,
        sms_last_hour=0,
        unique_dest_24h=3,
        timestamp=datetime(2024, 1, 15, 2, 0, 0),
    )


# ---------------------------------------------------------------------------
# Distribution fixtures for drift detection
# ---------------------------------------------------------------------------

@pytest.fixture
def reference_features() -> np.ndarray:
    """Reference feature distribution for drift detection tests."""
    rng = np.random.default_rng(seed=42)
    return rng.normal(loc=2.0, scale=0.5, size=1000)


@pytest.fixture
def drifted_features() -> np.ndarray:
    """Drifted feature distribution — significantly different from reference."""
    rng = np.random.default_rng(seed=42)
    return rng.normal(loc=8.0, scale=1.5, size=1000)


# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------

class FakeModelRegistry(IModelRegistry):
    """Fake model registry for unit tests — no pkl loading."""

    def __init__(
        self,
        fraud_proba: float = 0.94,
        version: str = "fake-v1.0.0",
    ) -> None:
        self._fraud_proba = fraud_proba
        self._version = version
        self._loaded = False

    def load(self, model_path: str) -> None:
        self._loaded = True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        legit_proba = 1.0 - self._fraud_proba
        return np.array([[legit_proba, self._fraud_proba]])

    def model_version(self) -> str:
        return self._version

    def feature_names(self) -> list[str]:
        return [
            "duration_normalized",
            "calls_last_hour",
            "is_night_call",
            "is_international",
            "sms_last_hour",
            "unique_dest_24h",
        ]


class FakeFeatureExtractor(IFeatureExtractor):
    """Fake feature extractor for unit tests."""

    def __init__(self, fixed_features: np.ndarray | None = None) -> None:
        self._features = fixed_features if fixed_features is not None else np.array(
            [[1.0, 45.0, 1.0, 1.0, 80.0, 50.0]]
        )

    def extract(self, event: CallEvent) -> FeatureVector:
        return FeatureVector(
            values=self._features,
            feature_names=[
                "duration_normalized",
                "calls_last_hour",
                "is_night_call",
                "is_international",
                "sms_last_hour",
                "unique_dest_24h",
            ],
            version="fake-v1",
        )


@pytest.fixture
def fake_registry() -> FakeModelRegistry:
    return FakeModelRegistry()


@pytest.fixture
def fake_extractor() -> FakeFeatureExtractor:
    return FakeFeatureExtractor()
