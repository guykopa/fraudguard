from fastapi import APIRouter
from pydantic import BaseModel

from fraudguard.domain.exceptions.exceptions import ModelNotLoadedError
from fraudguard.interfaces.i_model_registry import IModelRegistry

router = APIRouter()


class HealthResponse(BaseModel):
    status: str


def make_health_router(registry: IModelRegistry) -> APIRouter:
    """Factory returning health and readiness check routes."""

    @router.get("/health", response_model=HealthResponse)
    async def liveness() -> HealthResponse:
        return HealthResponse(status="ok")

    @router.get("/ready", response_model=HealthResponse)
    async def readiness() -> HealthResponse:
        try:
            registry.model_version()
            return HealthResponse(status="ready")
        except ModelNotLoadedError:
            return HealthResponse(status="not ready")

    return router
