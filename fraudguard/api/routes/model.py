from fastapi import APIRouter, Depends
from pydantic import BaseModel

from fraudguard.api.security.jwt_handler import decode_token, oauth2_scheme
from fraudguard.interfaces.i_model_registry import IModelRegistry

router = APIRouter()


class ModelInfoResponse(BaseModel):
    version: str
    feature_names: list[str]


def make_model_router(registry: IModelRegistry) -> APIRouter:
    """Factory returning a router with the model registry injected."""

    @router.get("/api/v1/model/info", response_model=ModelInfoResponse)
    async def model_info(
        token: str = Depends(oauth2_scheme),
    ) -> ModelInfoResponse:
        decode_token(token)
        return ModelInfoResponse(
            version=registry.model_version(),
            feature_names=registry.feature_names(),
        )

    return router
