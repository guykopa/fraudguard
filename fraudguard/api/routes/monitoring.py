from fastapi import APIRouter, Depends
from pydantic import BaseModel

from fraudguard.api.security.jwt_handler import decode_token, oauth2_scheme
from fraudguard.domain.services.drift_service import DriftService

router = APIRouter()


class DriftReportResponse(BaseModel):
    feature: str
    psi_score: float
    status: str
    reference_mean: float
    current_mean: float
    checked_at: str


def make_monitoring_router(drift_service: DriftService) -> APIRouter:
    """Factory returning a router with the drift service injected."""

    @router.get("/api/v1/monitoring/drift", response_model=list[DriftReportResponse])
    async def get_drift(
        token: str = Depends(oauth2_scheme),
    ) -> list[DriftReportResponse]:
        decode_token(token)
        reports = drift_service.compute_all()
        return [
            DriftReportResponse(
                feature=r.feature,
                psi_score=r.psi_score,
                status=r.status.value,
                reference_mean=r.reference_mean,
                current_mean=r.current_mean,
                checked_at=r.checked_at.isoformat(),
            )
            for r in reports
        ]

    return router
