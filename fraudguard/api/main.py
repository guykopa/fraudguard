import os

from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from fraudguard.adapters.sklearn_model_registry import SklearnModelRegistry
from fraudguard.api.routes.health import make_health_router
from fraudguard.api.routes.model import make_model_router
from fraudguard.api.routes.monitoring import make_monitoring_router
from fraudguard.api.routes.predict import make_predict_router
from fraudguard.api.security.jwt_handler import create_access_token
from fraudguard.domain.services.drift_service import DriftService
from fraudguard.domain.services.prediction_service import PredictionService
from fraudguard.ml.feature_engineering import PandasFeatureExtractor
from fraudguard.monitoring.drift_detector import PSIDriftDetector
from fraudguard.monitoring.metrics import PrometheusPublisher
from fraudguard.pipeline.inference_pipeline import InferencePipeline

MODEL_PATH = os.environ.get("MODEL_PATH", "models/fraud_detector_v1.pkl")
REFERENCE_PATH = os.environ.get("REFERENCE_PATH", "data/reference_dataset.csv")

# ---------------------------------------------------------------------------
# Wire concrete adapters — only place in the codebase that touches
# concrete classes directly (DIP composition root)
# ---------------------------------------------------------------------------
registry = SklearnModelRegistry()
registry.load(MODEL_PATH)

extractor = PandasFeatureExtractor()
publisher = PrometheusPublisher()
PrometheusPublisher.set_model_version(registry.model_version())

prediction_service = PredictionService(
    model_registry=registry,
    feature_extractor=extractor,
)
pipeline = InferencePipeline(
    prediction_service=prediction_service,
    metrics_publisher=publisher,
)

drift_detector = PSIDriftDetector()
drift_service = DriftService(
    drift_detector=drift_detector,
    reference_path=REFERENCE_PATH,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="fraudguard", version="1.0.0")

Instrumentator().instrument(app).expose(app)

app.include_router(make_health_router(registry))
app.include_router(make_predict_router(pipeline))
app.include_router(make_monitoring_router(drift_service))
app.include_router(make_model_router(registry))


# ---------------------------------------------------------------------------
# Auth token endpoint
# ---------------------------------------------------------------------------
class TokenResponse(BaseModel):
    access_token: str
    token_type: str


_DEMO_USERS = {"admin": "secret"}

auth_router = APIRouter()


@auth_router.post("/auth/token", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    password = _DEMO_USERS.get(form.username)
    if password is None or password != form.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
        )
    token = create_access_token(subject=form.username)
    return TokenResponse(access_token=token, token_type="bearer")


app.include_router(auth_router)
