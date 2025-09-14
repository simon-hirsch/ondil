from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Log
from ..types import ParameterShapes


class LogNormalMedian(ScipyMixin, Distribution):
    """
    The Log-Normal distribution with median and standard deviation parameterization in the log-space.

    The probability density function of the distribution is defined as:
    $$
    f(y | \\mu, \\sigma) = \\frac{1}{y\\sigma\\sqrt{2\\pi}} \\exp\\left(-\\frac{(\\log y - \\log \\mu)^2}{2\\sigma^2}\\right).
    $$
    respectively
    $$
    f(y | \\theta_0, \\theta_1) = \\frac{1}{y\\theta_1\\sqrt{2\\pi}}\\exp\\left(-\\frac{(\\log y - \\log \\theta_0)^2}{2\\theta_1^2}\\right).
    $$
    where $y$ is the observed data, $\\mu = \\theta_0$ is the median parameter and $\\sigma = \\theta_1$ is the scale parameter.
    """

    corresponding_gamlss: str = "LOGNO2"
    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (np.nextafter(0, 1), np.inf),
        1: (np.nextafter(0, 1), np.inf),
    }
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }
    distribution_support = (0, np.inf)
    scipy_dist = st.lognorm
    scipy_names = {"mu": "scale", "sigma": "s"}

    def __init__(
        self,
        loc_link: LinkFunction = Log(),
        scale_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
            }
        )

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        mu, sigma = self.theta_to_params(theta)
        return {"s": sigma, "scale": mu, "loc": 0}

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            return (np.log(y) - np.log(mu)) / (mu * sigma**2)
        elif param == 1:
            return ((np.log(y) - np.log(mu)) ** 2 - sigma**2) / (sigma**3)

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            return -1 / (mu**2 * sigma**2)
        elif param == 1:
            return -2 / (sigma**2)

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        return np.zeros_like(y)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        log_y = np.log(y)
        out = np.zeros((y.shape[0], self.n_params))
        out[:, 0] = np.exp(np.median(log_y, axis=0))
        out[:, 1] = np.std(log_y, axis=0)
        return out
