from abc import ABC, abstractmethod
from typing import Tuple


class DiagnosticDisplay(ABC):
    @classmethod
    @abstractmethod
    def from_estimator(
        cls, estimator, X, y, ax=None, figsize: Tuple[float, float] = (10, 5), **kwargs
    ):
        raise NotImplementedError("Should implement from_estimator method")

    @abstractmethod
    def plot(self, ax=None, figsize: Tuple[float, float] = (10, 5), **kwargs):
        raise NotImplementedError("Should implement plot method")
