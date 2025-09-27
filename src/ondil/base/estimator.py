from abc import ABC

import numpy as np
from sklearn.utils.validation import check_is_fitted


class OndilEstimatorMixin(ABC):
    @property
    def is_fitted(self) -> bool:
        """Has the estimator been fitted."""
        return hasattr(self, "n_observations_")

    @property
    def n_samples_(self) -> int:
        check_is_fitted(self, "n_observations_")
        return self.n_observations_

    def _print_message(
        self,
        message: str,
        level: int = 0,
    ) -> None:
        """Print a message if the verbosity level is sufficient.

        Message will be printed if level <= self.verbose.

        Args:
            message (str): Message to print.
            level (int, optional): Verbosity level. Defaults to 0.
        """
        if level <= self.verbose:
            print(f"[{self.__class__.__name__}]", message)

    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ):
        """
        Align ondil with the scikit-learn API for partial fitting.

        The first partial fit will call `fit`, and subsequent calls will call `update`.
        Allows furthermore to use the sklearn testing framework.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target values.
        sample_weight : np.ndarray, optional
            Sample weights for the observations, by default None.
        Returns
        -------
        self : Estimator
            The fitted estimator.

        """
        if self.is_fitted:
            self.update(X=X, y=y, sample_weight=sample_weight)
        else:
            self.fit(X=X, y=y, sample_weight=sample_weight)
        return self


class Estimator(ABC):
    def __init__(self):
        self.n_observations: int
        self.n_training: float

    @property
    def is_fitted(self) -> bool:
        """Has the estimator been fitted."""
        return self.n_observations > 0

    @property
    def n_obs(self) -> int:
        return self.n_observations

    @property
    def n_train(self) -> float:
        return self.n_training

    @property
    def coef_(self):
        return self.beta

    @property
    def coef_path_(self):
        return self.beta_path

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        raise NotImplementedError("Not Implemented")

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        raise NotImplementedError("Not Implemented")

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.update(X=X, y=y, sample_weights=sample_weights)
