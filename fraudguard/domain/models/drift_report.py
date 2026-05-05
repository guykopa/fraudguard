from dataclasses import dataclass
from datetime import datetime
from fraudguard.domain.models.drift_status import DriftStatus


@dataclass(frozen=True)
class DriftReport:
    """Immutable PSI drift analysis result for a single feature."""

    feature: str
    psi_score: float
    status: DriftStatus
    reference_mean: float
    current_mean: float
    checked_at: datetime
