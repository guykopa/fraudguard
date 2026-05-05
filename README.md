# fraudguard

Production-grade ML inference service for real-time telecom fraud detection.

## Stack

Python 3.11 · scikit-learn · FastAPI · Prometheus · Docker

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train model (one-time)
python -m fraudguard.ml.model_trainer

# Run API
JWT_SECRET=your-secret uvicorn fraudguard.api.main:app --reload
```

## API

```bash
# Get token
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=secret"

# Predict
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "msisdn": "+33612345678",
    "duration_sec": 3600,
    "destination": "+9XXXXXXXXX",
    "call_type": "INTERNATIONAL",
    "hour_of_day": 3,
    "calls_last_hour": 45,
    "sms_last_hour": 80,
    "unique_dest_24h": 50,
    "timestamp": "2024-01-15T03:15:00"
  }'
```

## Tests

```bash
JWT_SECRET=test-secret pytest tests/ --cov=fraudguard --cov-fail-under=90
```

## Docker

```bash
JWT_SECRET=your-secret docker-compose -f docker/docker-compose.yml up
```

Prometheus: http://localhost:9090 · Grafana: http://localhost:3000 (admin/admin)
