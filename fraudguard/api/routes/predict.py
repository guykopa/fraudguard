from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from fraudguard.api.security.jwt_handler import decode_token, oauth2_scheme
from fraudguard.domain.models.call_event import CallEvent
from fraudguard.pipeline.inference_pipeline import InferencePipeline

router = APIRouter()


class CallEventRequest(BaseModel):
    msisdn: str
    duration_sec: float
    destination: str
    call_type: str
    hour_of_day: int
    calls_last_hour: int
    sms_last_hour: int
    unique_dest_24h: int
    timestamp: datetime


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    risk_score: float
    model_version: str
    inference_latency_ms: float


def make_predict_router(pipeline: InferencePipeline) -> APIRouter:
    """Factory returning a router with the pipeline injected."""

    @router.post("/api/v1/predict", response_model=PredictionResponse)
    async def predict(
        request: CallEventRequest,
        token: str = Depends(oauth2_scheme),
    ) -> PredictionResponse:
        decode_token(token)

        event = CallEvent(
            msisdn=request.msisdn,
            duration_sec=request.duration_sec,
            destination=request.destination,
            call_type=request.call_type,
            hour_of_day=request.hour_of_day,
            calls_last_hour=request.calls_last_hour,
            sms_last_hour=request.sms_last_hour,
            unique_dest_24h=request.unique_dest_24h,
            timestamp=request.timestamp,
        )

        prediction = pipeline.run(event)

        return PredictionResponse(
            label=prediction.label.value,
            confidence=prediction.confidence,
            risk_score=prediction.risk_score,
            model_version=prediction.model_version,
            inference_latency_ms=prediction.inference_latency_ms,
        )

    return router
