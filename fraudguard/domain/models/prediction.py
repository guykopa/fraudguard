from dataclasses import dataclass
from fraudguard.domain.models.fraud_label import FraudLabel


@dataclass(frozen=True)
class Prediction:
    """Immutable result of a fraud inference for a single CallEvent."""

    label: FraudLabel
    confidence: float
    risk_score: float
    model_version: str
    inference_latency_ms: float
