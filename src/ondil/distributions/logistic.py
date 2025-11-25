from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Identity, Log
from ..types import ParameterShapes


class Logistic(ScipyMixin, Distribution):
    r"""
    The Logistic distribution with location and scale parameterization.

    The probability density function is:
    $$
    f(y | \\mu, \\sigma) = \\frac{\\exp\\left(-\\frac{y - \\mu}{\\sigma}\\right)}{\\sigma \\left(1 + \\exp\\left(-\\frac{y - \\mu}{\\sigma}\\right)\\right)^2}
    $$

    This distribution corresponds to the LO() distribution in GAMLSS.
    """

    corresponding_gamlss: str = "LO"
    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
    }
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }
    distribution_support = (-np.inf, np.inf)
    scipy_dist = st.logistic
    scipy_names = {"mu": "loc", "sigma": "scale"}

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(links={0: loc_link, 1: scale_link})

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        mu = theta[:, 0]
        sigma = theta[:, 1]
        return {"loc": mu, "scale": sigma}

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)
        z = (y - mu) / sigma
        expz = np.exp(z)

        if param == 0:
            return (1 / sigma) * (expz - 1) / (1 + expz)

        if param == 1:
            return (
                -1 / sigma
                - (y - mu) / sigma**2
                + 2 * ((y - mu) / sigma**2) * expz / (1 + expz)
            )

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        _, sigma = self.theta_to_params(theta)

        if param == 0:
            return -1 / (3 * sigma**2)

        if param == 1:
            return -(1 / (3 * sigma**2)) * (1 + (np.pi**2) / 3)

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        return np.zeros_like(y)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        initial_params = [np.mean(y), (np.sqrt(3) * np.std(y, ddof=1)) / np.sqrt(np.pi)]
        return np.tile(initial_params, (y.shape[0], 1))
