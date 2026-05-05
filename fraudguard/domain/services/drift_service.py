import numpy as np
import pandas as pd

from fraudguard.domain.models.drift_report import DriftReport
from fraudguard.interfaces.i_drift_detector import IDriftDetector


class DriftService:
    """Computes PSI drift reports for all features against the reference dataset."""

    def __init__(
        self,
        drift_detector: IDriftDetector,
        reference_path: str,
        current_data: pd.DataFrame | None = None,
    ) -> None:
        self._detector = drift_detector
        self._reference_path = reference_path
        self._current_data = current_data

    def compute_all(self) -> list[DriftReport]:
        """Compute drift for every feature in the reference dataset.

        Returns:
            List of DriftReport, one per feature.
        """
        reference_df = pd.read_csv(self._reference_path)
        current_df = self._current_data if self._current_data is not None else reference_df

        reports: list[DriftReport] = []
        for feature in reference_df.columns:
            ref_array: np.ndarray = reference_df[feature].values
            cur_array: np.ndarray = current_df[feature].values
            report = self._detector.compute_psi(ref_array, cur_array, feature)
            reports.append(report)

        return reports
