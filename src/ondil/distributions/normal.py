from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Identity, Log
from ..types import ParameterShapes


class Normal(ScipyMixin, Distribution):
    """
    The Normal distribution with mean and standard deviation parameterization.

    The probability density function of the distribution is defined as:
    $$
        f(y | \\mu, \\sigma) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \exp\\left(-\\frac{(y - \\mu)^2}{2\\sigma^2}\\right).
    $$
    respectively
    $$
        f(y | \\theta_0, \\theta_1) = \\frac{1}{\\sqrt{2\\pi\\theta_1^2}} \exp\\left(-\\frac{(y - \\theta_0)^2}{2\\theta_1^2}\\right).
    $$
    where $y$ is the observed data, $\\mu = \\theta_0$ is the location parameter and $\\sigma = \\theta_1$ is the scale parameter.

    This distribution corresponds to the NO() distribution in GAMLSS.
    """

    corresponding_gamlss: str = "NO"
    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
    }
    distribution_support = (-np.inf, np.inf)
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }
    # Scipy equivalent and parameter mapping ondil -> scipy
    scipy_dist = st.norm
    scipy_names = {"mu": "loc", "sigma": "scale"}

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
    ) -> None:
        """Initialize the Normal.

        Args:
            loc_link (LinkFunction, optional): Location link. Defaults to Identity().
            scale_link (LinkFunction, optional): Scale link. Defaults to Log().
        """
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
            }
        )

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            return (1 / sigma**2) * (y - mu)

        if param == 1:
            return ((y - mu) ** 2 - sigma**2) / (sigma**3)

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        _, sigma = self.theta_to_params(theta)
        if param == 0:
            # MU
            return -(1 / sigma**2)

        if param == 1:
            # SIGMA
            return -(2 / (sigma**2))

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        if sorted(params) == [0, 1]:
            return np.zeros_like(y)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        initial_params = [np.mean(y), np.std(y, ddof=1)]
        return np.tile(initial_params, (y.shape[0], 1))


class NormalMeanVariance(ScipyMixin, Distribution):
    """
    The Normal distribution with mean and variance parameterization.

    The probability density function of the distribution is defined as:
    $$
        f(y | \\mu, \\sigma^2) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \exp\\left(-\\frac{(y - \\mu)^2}{2\\sigma^2}\\right).
    $$
    respectively
    $$
        f(y | \\theta_0, \\theta_1) = \\frac{1}{\\sqrt{2\\pi\\theta_1}} \exp\\left(-\\frac{(y - \\theta_0)^2}{2\\theta_1}\\right).
    $$
    where $y$ is the observed data, $\\mu = \\theta_0$ is the location parameter and $\\sigma^2 = \\theta_1$ is the scale parameter.
    """

    corresponding_gamlss: str = "NO2"
    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
    }
    distribution_support = (-np.inf, np.inf)
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }
    # Scipy equivalent and parameter mapping ondil -> scipy
    scipy_dist = st.norm
    scipy_names = {"mu": "loc", "sigma": "scale"}

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
    ) -> None:
        """Initialize the NormalMeanVariance.

        Args:
            loc_link (LinkFunction, optional): Location link. Defaults to Identity().
            scale_link (LinkFunction, optional): Scale link. Defaults to Log().
        """
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
            }
        )

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        """Map GAMLSS Parameters to scipy parameters.

        Args:
            theta (np.ndarray): parameters

        Returns:
            dict: Dict of (loc, scale) for scipy.stats.norm(loc, scale)
        """
        mu = theta[:, 0]
        sigma = theta[:, 1]
        params = {"loc": mu, "scale": sigma**0.5}
        return params

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            return (1 / sigma) * (y - mu)

        if param == 1:
            return 0.5 * ((y - mu) ** 2 - sigma) / (sigma**2)

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        _, sigma = self.theta_to_params(theta)
        if param == 0:
            # MU
            return -(1 / sigma)

        if param == 1:
            # SIGMA
            return -(1 / (2 * sigma**2))

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        if sorted(params) == [0, 1]:
            return np.zeros_like(y)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        initial_params = [np.mean(y), np.var(y, ddof=1)]
        return np.tile(initial_params, (y.shape[0], 1))
