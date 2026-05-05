"""One-time script: generates synthetic telecom data, trains and saves the model."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


FEATURE_NAMES = [
    "duration_normalized",
    "calls_last_hour",
    "is_night_call",
    "is_international",
    "sms_last_hour",
    "unique_dest_24h",
]

MODEL_PATH = "models/fraud_detector_v1.pkl"
REFERENCE_PATH = "data/reference_dataset.csv"
N_SAMPLES = 10_000
FRAUD_RATIO = 0.20


def _generate_data(rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic telecom call data with realistic fraud patterns."""
    n_fraud = int(N_SAMPLES * FRAUD_RATIO)
    n_legit = N_SAMPLES - n_fraud

    legit = pd.DataFrame({
        "duration_normalized": rng.uniform(0.01, 0.3, n_legit),
        "calls_last_hour": rng.integers(0, 8, n_legit).astype(float),
        "is_night_call": rng.choice([0, 1], n_legit, p=[0.85, 0.15]).astype(float),
        "is_international": rng.choice([0, 1], n_legit, p=[0.9, 0.1]).astype(float),
        "sms_last_hour": rng.integers(0, 10, n_legit).astype(float),
        "unique_dest_24h": rng.integers(1, 15, n_legit).astype(float),
        "label": 0,
    })

    fraud = pd.DataFrame({
        "duration_normalized": rng.uniform(0.5, 1.0, n_fraud),
        "calls_last_hour": rng.integers(20, 60, n_fraud).astype(float),
        "is_night_call": rng.choice([0, 1], n_fraud, p=[0.3, 0.7]).astype(float),
        "is_international": rng.choice([0, 1], n_fraud, p=[0.2, 0.8]).astype(float),
        "sms_last_hour": rng.integers(30, 100, n_fraud).astype(float),
        "unique_dest_24h": rng.integers(20, 60, n_fraud).astype(float),
        "label": 1,
    })

    return pd.concat([legit, fraud], ignore_index=True).sample(
        frac=1, random_state=42
    )


def train() -> None:
    """Train RandomForestClassifier and persist artifacts."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    rng = np.random.default_rng(seed=42)
    df = _generate_data(rng)

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "version": "v1.0.0"}, MODEL_PATH)
    df[FEATURE_NAMES].to_csv(REFERENCE_PATH, index=False)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Reference dataset saved to {REFERENCE_PATH}")


if __name__ == "__main__":
    train()
