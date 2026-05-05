from abc import ABC, abstractmethod
import numpy as np
from fraudguard.domain.models.drift_report import DriftReport


class IDriftDetector(ABC):
    """Contract for detecting model input drift using PSI."""

    @abstractmethod
    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str,
    ) -> DriftReport:
        """Compute Population Stability Index between two distributions.

        Args:
            reference: Reference distribution (training baseline).
            current: Current production distribution.
            feature_name: Name of the feature being monitored.

        Returns:
            DriftReport with PSI score and drift status.
        """
