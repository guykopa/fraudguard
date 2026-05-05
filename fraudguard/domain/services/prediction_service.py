import time

from fraudguard.domain.models.call_event import CallEvent
from fraudguard.domain.models.prediction import Prediction
from fraudguard.domain.models.fraud_label import FraudLabel
from fraudguard.interfaces.i_model_registry import IModelRegistry
from fraudguard.interfaces.i_feature_extractor import IFeatureExtractor

_FRAUD_THRESHOLD = 0.7
_SUSPICIOUS_THRESHOLD = 0.4


class PredictionService:
    """Orchestrates the inference pipeline: CallEvent → Prediction.

    Depends only on ports — never on sklearn or pandas directly.
    """

    def __init__(
        self,
        model_registry: IModelRegistry,
        feature_extractor: IFeatureExtractor,
    ) -> None:
        self._registry = model_registry
        self._extractor = feature_extractor

    def predict(self, event: CallEvent) -> Prediction:
        """Run the full inference pipeline for a single call event.

        Args:
            event: Raw telecom call event.

        Returns:
            Prediction with label, scores, version and latency.
        """
        start = time.perf_counter()

        feature_vector = self._extractor.extract(event)
        probas = self._registry.predict_proba(feature_vector.values)
        risk_score = float(probas[0][1])

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        label = self._classify(risk_score)

        return Prediction(
            label=label,
            confidence=max(float(probas[0][0]), float(probas[0][1])),
            risk_score=risk_score,
            model_version=self._registry.model_version(),
            inference_latency_ms=elapsed_ms,
        )

    @staticmethod
    def _classify(risk_score: float) -> FraudLabel:
        if risk_score >= _FRAUD_THRESHOLD:
            return FraudLabel.FRAUD
        if risk_score >= _SUSPICIOUS_THRESHOLD:
            return FraudLabel.SUSPICIOUS
        return FraudLabel.LEGITIMATE
