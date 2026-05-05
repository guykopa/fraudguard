from abc import ABC, abstractmethod
import numpy as np


class IModelRegistry(ABC):
    """Contract for loading and using a trained ML model."""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load a serialized model from disk.

        Args:
            model_path: Path to the .pkl file.

        Raises:
            ModelNotFoundError: if the file does not exist.
        """

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities for input features.

        Args:
            features: Feature array of shape (1, n_features).

        Returns:
            Probability array of shape (1, n_classes).
        """

    @abstractmethod
    def model_version(self) -> str:
        """Return the version string of the loaded model."""

    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return ordered list of feature names the model expects."""
