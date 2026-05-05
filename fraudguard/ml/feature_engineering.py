import numpy as np
import pandas as pd

from fraudguard.domain.models.call_event import CallEvent
from fraudguard.domain.models.feature_vector import FeatureVector
from fraudguard.interfaces.i_feature_extractor import IFeatureExtractor

_FEATURE_NAMES = [
    "duration_normalized",
    "calls_last_hour",
    "is_night_call",
    "is_international",
    "sms_last_hour",
    "unique_dest_24h",
]

_NIGHT_HOURS = {0, 1, 2, 3, 4, 5}
_VERSION = "v1.0.0"


class PandasFeatureExtractor(IFeatureExtractor):
    """Extracts the 6 fraud-detection features from a CallEvent using pandas."""

    def extract(self, event: CallEvent) -> FeatureVector:
        """Extract feature vector from a raw telecom call event.

        Args:
            event: Raw CallEvent with all input fields.

        Returns:
            FeatureVector with shape (1, 6) and ordered feature names.
        """
        row = {
            "duration_normalized": event.duration_sec / 3600.0,
            "calls_last_hour": float(event.calls_last_hour),
            "is_night_call": 1.0 if event.hour_of_day in _NIGHT_HOURS else 0.0,
            "is_international": 0.0 if event.destination.startswith("+33") else 1.0,
            "sms_last_hour": float(event.sms_last_hour),
            "unique_dest_24h": float(event.unique_dest_24h),
        }
        df = pd.DataFrame([row], columns=_FEATURE_NAMES)
        values: np.ndarray = df.values.astype(np.float64)
        return FeatureVector(values=values, feature_names=_FEATURE_NAMES, version=_VERSION)
