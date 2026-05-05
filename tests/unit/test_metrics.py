from fraudguard.domain.models.fraud_label import FraudLabel
from fraudguard.monitoring.metrics import PrometheusPublisher


class TestPrometheusPublisher:

    def setup_method(self) -> None:
        self.publisher = PrometheusPublisher()

    def test_publish_prediction_does_not_raise(self) -> None:
        self.publisher.publish_prediction(FraudLabel.FRAUD, latency_ms=12.5)

    def test_publish_prediction_legitimate_does_not_raise(self) -> None:
        self.publisher.publish_prediction(FraudLabel.LEGITIMATE, latency_ms=5.0)

    def test_publish_drift_does_not_raise(self) -> None:
        self.publisher.publish_drift("calls_last_hour", psi_score=0.05)

    def test_set_model_version_does_not_raise(self) -> None:
        PrometheusPublisher.set_model_version("v1.0.0")
