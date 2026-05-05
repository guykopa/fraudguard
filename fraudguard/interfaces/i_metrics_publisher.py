from abc import ABC, abstractmethod
from fraudguard.domain.models.fraud_label import FraudLabel


class IMetricsPublisher(ABC):
    """Contract for publishing prediction and drift metrics."""

    @abstractmethod
    def publish_prediction(self, label: FraudLabel, latency_ms: float) -> None:
        """Publish prediction outcome and inference latency.

        Args:
            label: Fraud classification result.
            latency_ms: Inference duration in milliseconds.
        """

    @abstractmethod
    def publish_drift(self, feature: str, psi_score: float) -> None:
        """Publish PSI drift score for a monitored feature.

        Args:
            feature: Name of the feature.
            psi_score: Computed PSI value.
        """
