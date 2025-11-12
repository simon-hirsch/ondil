from abc import ABC, abstractmethod

import numpy as np


class Term(ABC):
    """The base class for terms in structured additive distributional regression."""

    def _prepare_term(self) -> "Term":
        return self

    @property
    @abstractmethod
    def allow_online_updates(self) -> bool:
        return True

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
