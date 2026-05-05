FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraudguard/ fraudguard/
COPY models/ models/
COPY data/ data/

ENV JWT_SECRET=change-me-in-production
ENV MODEL_PATH=models/fraud_detector_v1.pkl
ENV REFERENCE_PATH=data/reference_dataset.csv

EXPOSE 8000

CMD ["uvicorn", "fraudguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
