from abc import ABC, abstractmethod
from fraudguard.domain.models.call_event import CallEvent
from fraudguard.domain.models.feature_vector import FeatureVector


class IFeatureExtractor(ABC):
    """Contract for extracting ML features from a CallEvent."""

    @abstractmethod
    def extract(self, event: CallEvent) -> FeatureVector:
        """Extract feature vector from a raw telecom call event.

        Args:
            event: Raw CallEvent with all input fields.

        Returns:
            FeatureVector with numpy array and feature names.

        Raises:
            FeatureExtractionError: if required fields are missing.
        """
