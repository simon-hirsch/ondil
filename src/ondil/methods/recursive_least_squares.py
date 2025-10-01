from typing import Optional

import numpy as np

from ..base import EstimationMethod
from ..gram import init_inverted_gram, init_y_gram, update_inverted_gram, update_y_gram


class OrdinaryLeastSquares(EstimationMethod):
    """Simple ordinary least squares respectively recursive least squares. No fancy parameters possible."""

    def __init__(self) -> None:
        super().__init__(
            _path_based_method=False,
            _accepts_bounds=False,
            _accepts_selection=False,
        )

    @staticmethod
    def init_x_gram(X: np.ndarray, weights: np.ndarray, forget: float) -> np.ndarray:
        return init_inverted_gram(X, w=weights, forget=forget)

    @staticmethod
    def init_y_gram(
        X: np.ndarray, y: np.ndarray, weights: np.ndarray, forget: float
    ) -> np.ndarray:
        return init_y_gram(X, y, w=weights, forget=forget)

    @staticmethod
    def update_x_gram(
        gram: np.ndarray, X: np.ndarray, weights: np.ndarray, forget: float
    ) -> np.ndarray:
        return update_inverted_gram(gram, X, w=weights, forget=forget)

    @staticmethod
    def update_y_gram(
        gram: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        forget: float,
    ) -> np.ndarray:
        return update_y_gram(gram, X, y, forget=forget, w=weights)

    def fit_beta_path(
        self, x_gram: np.ndarray, y_gram: np.ndarray, is_regularized: np.ndarray
    ) -> np.ndarray:
        return super().fit_beta_path(x_gram, y_gram, is_regularized)

    def update_beta_path(
        self,
        x_gram: np.ndarray,
        y_gram: np.ndarray,
        beta_path: np.ndarray,
        is_regularized: np.ndarray,
    ) -> np.ndarray:
        return super().update_beta_path(x_gram, y_gram, beta_path, is_regularized)

    def fit_beta(
        self,
        x_gram: np.ndarray,
        y_gram: np.ndarray,
        is_regularized: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return (x_gram @ y_gram).squeeze(-1)

    def update_beta(
        self,
        x_gram: np.ndarray,
        y_gram: np.ndarray,
        beta: np.ndarray,
        is_regularized: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return (x_gram @ y_gram).squeeze(-1)
