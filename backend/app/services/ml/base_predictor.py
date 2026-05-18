"""Abstract base class for all model predictors."""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """All model-specific predictors inherit from this class."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Run inference.
        Args:
            features: DataFrame of normalised features, shape (N, 10)
        Returns:
            np.ndarray of normalised predictions, shape (N,)
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...
