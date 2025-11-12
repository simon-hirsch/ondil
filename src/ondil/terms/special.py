from typing import Literal

import numpy as np

from ..base import Term
from ..design_matrix import subset_array


class ScikitLearnEstimatorTerm(Term):
    """Linear term for structured additive distributional regression."""

    allow_online_updates: bool = False

    def __init__(
        self,
        sklearn_estimator,
        features: np.ndarray | list[int] | Literal["all"],
    ):
        self.sklearn_estimator = sklearn_estimator
        self.features = features

    def _prepare_term(self):
        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "ScikitLearnEstimatorTerm":
        X_mat = subset_array(X, self.features)

        if sample_weight is not None:
            self.sklearn_estimator.fit(X_mat, y, sample_weight=sample_weight)
        else:
            self.sklearn_estimator.fit(X_mat, y)
        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        X_mat = subset_array(X, self.features)
        return self.sklearn_estimator.predict(X_mat)

    def update(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        sample_weight: np.ndarray = None,
    ) -> "ScikitLearnEstimatorTerm":
        raise NotImplementedError(
            "Online updates are not supported for ScikitLearnEstimatorTerm."
        )
