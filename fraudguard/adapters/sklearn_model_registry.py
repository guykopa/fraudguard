import os
import joblib
import numpy as np

from fraudguard.interfaces.i_model_registry import IModelRegistry
from fraudguard.domain.exceptions.exceptions import ModelNotFoundError, ModelNotLoadedError


class SklearnModelRegistry(IModelRegistry):
    """Loads and serves a scikit-learn model from a joblib-serialized .pkl file."""

    def __init__(self) -> None:
        self._model = None
        self._version: str = ""
        self._feature_names: list[str] = []

    def load(self, model_path: str) -> None:
        """Load a serialized model artifact from disk.

        Args:
            model_path: Path to the joblib .pkl file.

        Raises:
            ModelNotFoundError: if the file does not exist.
        """
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model file not found: {model_path}")

        artifact = joblib.load(model_path)
        self._model = artifact["model"]
        self._version = artifact["version"]
        self._feature_names = artifact["feature_names"]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities for input features.

        Args:
            features: Feature array of shape (1, n_features).

        Returns:
            Probability array of shape (1, n_classes).

        Raises:
            ModelNotLoadedError: if load() has not been called.
        """
        if self._model is None:
            raise ModelNotLoadedError("Model not loaded. Call load() first.")
        return self._model.predict_proba(features)

    def model_version(self) -> str:
        if self._model is None:
            raise ModelNotLoadedError("Model not loaded. Call load() first.")
        return self._version

    def feature_names(self) -> list[str]:
        if self._model is None:
            raise ModelNotLoadedError("Model not loaded. Call load() first.")
        return self._feature_names
