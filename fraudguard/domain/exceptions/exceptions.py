class ModelNotFoundError(FileNotFoundError):
    """Raised when the model file does not exist at the given path."""


class ModelNotLoadedError(RuntimeError):
    """Raised when predict is called before load()."""


class FeatureExtractionError(ValueError):
    """Raised when a CallEvent is missing required fields for feature extraction."""


class PredictionError(RuntimeError):
    """Raised when the inference pipeline fails unexpectedly."""
