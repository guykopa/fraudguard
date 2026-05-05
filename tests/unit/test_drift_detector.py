import numpy as np
from fraudguard.domain.models.drift_status import DriftStatus
from fraudguard.domain.models.drift_report import DriftReport


class TestDriftDetector:

    def setup_method(self) -> None:
        from fraudguard.monitoring.drift_detector import PSIDriftDetector
        self.detector = PSIDriftDetector()

    def test_no_drift_on_identical_distributions(
        self, reference_features: np.ndarray
    ) -> None:
        report = self.detector.compute_psi(
            reference_features, reference_features, "calls_last_hour"
        )
        assert report.status == DriftStatus.OK
        assert report.psi_score < 0.1

    def test_critical_drift_on_very_different_distributions(
        self,
        reference_features: np.ndarray,
        drifted_features: np.ndarray,
    ) -> None:
        report = self.detector.compute_psi(
            reference_features, drifted_features, "calls_last_hour"
        )
        assert report.status == DriftStatus.CRITICAL
        assert report.psi_score >= 0.2

    def test_report_contains_feature_name(
        self, reference_features: np.ndarray
    ) -> None:
        report = self.detector.compute_psi(
            reference_features, reference_features, "duration_normalized"
        )
        assert report.feature == "duration_normalized"

    def test_report_is_drift_report_dataclass(
        self, reference_features: np.ndarray
    ) -> None:
        report = self.detector.compute_psi(
            reference_features, reference_features, "sms_last_hour"
        )
        assert isinstance(report, DriftReport)

    def test_handles_zero_division_gracefully(self) -> None:
        reference = np.ones(100)
        current = np.zeros(100)
        report = self.detector.compute_psi(reference, current, "test_feature")
        assert report is not None
        assert isinstance(report.psi_score, float)

    def test_warning_drift_on_moderately_different_distributions(self) -> None:
        rng = np.random.default_rng(seed=0)
        reference = rng.normal(loc=2.0, scale=0.5, size=1000)
        moderate = rng.normal(loc=2.8, scale=0.6, size=1000)
        report = self.detector.compute_psi(reference, moderate, "unique_dest_24h")
        assert report.status in (DriftStatus.WARNING, DriftStatus.CRITICAL)

    def test_reference_mean_is_populated(
        self, reference_features: np.ndarray
    ) -> None:
        report = self.detector.compute_psi(
            reference_features, reference_features, "duration_normalized"
        )
        assert report.reference_mean == float(np.mean(reference_features))

    def test_current_mean_is_populated(
        self, reference_features: np.ndarray, drifted_features: np.ndarray
    ) -> None:
        report = self.detector.compute_psi(
            reference_features, drifted_features, "calls_last_hour"
        )
        assert report.current_mean == float(np.mean(drifted_features))
