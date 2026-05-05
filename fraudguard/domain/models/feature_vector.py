from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FeatureVector:
    """Immutable feature representation extracted from a CallEvent."""

    values: np.ndarray
    feature_names: list[str]
    version: str

    class Config:
        arbitrary_types_allowed = True
