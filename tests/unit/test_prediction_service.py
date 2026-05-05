import pytest
from tests.conftest import FakeModelRegistry, FakeFeatureExtractor
from fraudguard.domain.models.call_event import CallEvent
from fraudguard.domain.models.prediction import Prediction
from fraudguard.domain.models.fraud_label import FraudLabel


class TestPredictionService:

    def setup_method(self) -> None:
        from fraudguard.domain.services.prediction_service import PredictionService
        self.fake_registry = FakeModelRegistry(fraud_proba=0.94)
        self.fake_extractor = FakeFeatureExtractor()
        self.service = PredictionService(
            model_registry=self.fake_registry,
            feature_extractor=self.fake_extractor,
        )

    def test_returns_prediction_dataclass(self, fraud_call: CallEvent) -> None:
        result = self.service.predict(fraud_call)
        assert isinstance(result, Prediction)

    def test_returns_fraud_for_high_risk_score(self, fraud_call: CallEvent) -> None:
        result = self.service.predict(fraud_call)
        assert result.label == FraudLabel.FRAUD
        assert result.risk_score >= 0.7

    def test_legitimate_call_returns_legitimate_label(
        self, normal_call: CallEvent
    ) -> None:
        low_risk_registry = FakeModelRegistry(fraud_proba=0.05)
        from fraudguard.domain.services.prediction_service import PredictionService
        service = PredictionService(
            model_registry=low_risk_registry,
            feature_extractor=self.fake_extractor,
        )
        result = service.predict(normal_call)
        assert result.label == FraudLabel.LEGITIMATE

    def test_suspicious_label_for_mid_risk_score(self, normal_call: CallEvent) -> None:
        mid_risk_registry = FakeModelRegistry(fraud_proba=0.5)
        from fraudguard.domain.services.prediction_service import PredictionService
        service = PredictionService(
            model_registry=mid_risk_registry,
            feature_extractor=self.fake_extractor,
        )
        result = service.predict(normal_call)
        assert result.label == FraudLabel.SUSPICIOUS

    def test_model_version_propagated(self, fraud_call: CallEvent) -> None:
        result = self.service.predict(fraud_call)
        assert result.model_version == "fake-v1.0.0"

    def test_confidence_between_0_and_1(self, normal_call: CallEvent) -> None:
        result = self.service.predict(normal_call)
        assert 0.0 <= result.confidence <= 1.0

    def test_measures_inference_latency(self, normal_call: CallEvent) -> None:
        result = self.service.predict(normal_call)
        assert result.inference_latency_ms > 0

    def test_risk_score_equals_fraud_proba(self, fraud_call: CallEvent) -> None:
        result = self.service.predict(fraud_call)
        assert result.risk_score == pytest.approx(0.94)
