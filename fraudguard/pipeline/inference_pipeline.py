from fraudguard.domain.models.call_event import CallEvent
from fraudguard.domain.models.prediction import Prediction
from fraudguard.domain.services.prediction_service import PredictionService
from fraudguard.interfaces.i_metrics_publisher import IMetricsPublisher


class InferencePipeline:
    """Thin orchestrator: runs PredictionService and publishes metrics.

    Single entry point for the API layer — keeps routes free of
    business logic and metrics concerns.
    """

    def __init__(
        self,
        prediction_service: PredictionService,
        metrics_publisher: IMetricsPublisher,
    ) -> None:
        self._service = prediction_service
        self._publisher = metrics_publisher

    def run(self, event: CallEvent) -> Prediction:
        """Execute inference and publish outcome metrics.

        Args:
            event: Raw telecom call event.

        Returns:
            Prediction result.
        """
        prediction = self._service.predict(event)
        self._publisher.publish_prediction(prediction.label, prediction.inference_latency_ms)
        return prediction
