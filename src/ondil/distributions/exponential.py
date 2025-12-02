from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Log
from ..types import ParameterShapes


class Exponential(ScipyMixin, Distribution):
    r"""
    The Exponential distribution parameterized by the mean (mu).

    PDF:
    f(y | mu) = (1 / mu) * exp(-y / mu), for y > 0, mu > 0

    This corresponds to EXP() in GAMLSS where:
    - mu > 0
    - y > 0

    """

    corresponding_gamlss: str = "EXP"
    parameter_names = {0: "mu"}
    parameter_support = {0: (np.nextafter(0, 1), np.inf)}
    distribution_support = (np.nextafter(0, 1), np.inf)
    scipy_dist = st.expon
    scipy_names = {"mu": "scale"}
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }

    def __init__(self, mu_link: LinkFunction = Log()) -> None:
        assert isinstance(mu_link, LinkFunction), "mu_link must be a LinkFunction"
        super().__init__(links={0: mu_link})

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        (mu,) = self.theta_to_params(theta)
        return {"scale": mu}

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        (mu,) = self.theta_to_params(theta)
        return (y - mu) / mu**2

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        (mu,) = self.theta_to_params(theta)
        return -1 / mu**2

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        return np.zeros_like(y)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        return np.full((y.shape[0], 1), np.mean(y))
