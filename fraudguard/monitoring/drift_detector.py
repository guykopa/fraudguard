from datetime import datetime

import numpy as np

from fraudguard.domain.models.drift_report import DriftReport
from fraudguard.domain.models.drift_status import DriftStatus
from fraudguard.interfaces.i_drift_detector import IDriftDetector

_N_BINS = 10
_PSI_WARNING = 0.1
_PSI_CRITICAL = 0.2
_EPSILON = 1e-8


class PSIDriftDetector(IDriftDetector):
    """Computes Population Stability Index to detect feature distribution drift."""

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str,
    ) -> DriftReport:
        """Compute PSI between reference and current distributions.

        Args:
            reference: Reference distribution (training baseline).
            current: Current production distribution.
            feature_name: Name of the feature being monitored.

        Returns:
            DriftReport with PSI score, status and distribution means.
        """
        psi_score = self._psi(reference, current)
        status = self._classify(psi_score)

        return DriftReport(
            feature=feature_name,
            psi_score=psi_score,
            status=status,
            reference_mean=float(np.mean(reference)),
            current_mean=float(np.mean(current)),
            checked_at=datetime.utcnow(),
        )

    def _psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate PSI between two arrays using shared bin edges."""
        all_values = np.concatenate([reference, current])
        bin_edges = np.percentile(all_values, np.linspace(0, 100, _N_BINS + 1))

        # Ensure unique edges to avoid empty bins
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        ref_pct = ref_counts / (len(reference) + _EPSILON)
        cur_pct = cur_counts / (len(current) + _EPSILON)

        # Replace zeros to avoid log(0)
        ref_pct = np.where(ref_pct == 0, _EPSILON, ref_pct)
        cur_pct = np.where(cur_pct == 0, _EPSILON, cur_pct)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return round(psi, 6)

    @staticmethod
    def _classify(psi_score: float) -> DriftStatus:
        if psi_score >= _PSI_CRITICAL:
            return DriftStatus.CRITICAL
        if psi_score >= _PSI_WARNING:
            return DriftStatus.WARNING
        return DriftStatus.OK
