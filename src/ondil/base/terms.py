from abc import ABC, abstractmethod

import numpy as np


class Term(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "Term":
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "Term":
        raise NotImplementedError("Not implemented")
