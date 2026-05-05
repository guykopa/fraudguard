from prometheus_client import Counter, Histogram, Gauge, Info

from fraudguard.domain.models.fraud_label import FraudLabel
from fraudguard.interfaces.i_metrics_publisher import IMetricsPublisher

_predictions_total = Counter(
    "fraudguard_predictions_total",
    "Total number of predictions",
    ["label"],
)

_inference_latency = Histogram(
    "fraudguard_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

_drift_psi_score = Gauge(
    "fraudguard_drift_psi_score",
    "PSI drift score per feature",
    ["feature"],
)

_model_version_info = Info(
    "fraudguard_model",
    "Model version metadata",
)


class PrometheusPublisher(IMetricsPublisher):
    """Publishes prediction and drift metrics to Prometheus."""

    def publish_prediction(self, label: FraudLabel, latency_ms: float) -> None:
        """Increment prediction counter and record latency.

        Args:
            label: Fraud classification result.
            latency_ms: Inference duration in milliseconds.
        """
        _predictions_total.labels(label=label.value).inc()
        _inference_latency.observe(latency_ms / 1000.0)

    def publish_drift(self, feature: str, psi_score: float) -> None:
        """Update PSI gauge for a monitored feature.

        Args:
            feature: Name of the feature.
            psi_score: Computed PSI value.
        """
        _drift_psi_score.labels(feature=feature).set(psi_score)

    @staticmethod
    def set_model_version(version: str) -> None:
        """Register model version in Prometheus info metric."""
        _model_version_info.info({"version": version})
