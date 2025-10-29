from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Log
from ..types import ParameterShapes


class Poisson(ScipyMixin, Distribution):
    """The Poisson Distribution for GAMLSS.

    The distribution function is defined as in GAMLSS as:
    $$
    f(y|\\mu)=\\frac{e^{-\\mu} \\mu^y}{y!}
    $$

    with the location parameter $\\mu > 0$.

    !!! Note
        The function is parameterized as GAMLSS' PO() distribution.

        This parameterization matches the `scipy.stats.poisson(mu)` parameterization.

    The `scipy.stats.poisson()` distribution is defined as:
    $$
    f(k, \\mu) = \\frac{e^{-\\mu} \\mu^k}{k!}
    $$
    with the parameter $\\mu >0$.

    Args:
        loc_link (LinkFunction, optional): The link function for $\\mu$. Defaults to Log().
    """

    corresponding_gamlss: str = "PO"

    parameter_names = {0: "mu"}
    parameter_support = {
        0: (np.nextafter(0, 1), np.inf),
    }
    parameter_shape = {
        0: ParameterShapes.SCALAR,
    }
    distribution_support = (0, np.inf)
    # Scipy equivalent and parameter mapping ondil -> scipy
    scipy_dist = st.poisson
    scipy_names = {"mu": "mu"}

    def __init__(
        self,
        loc_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(links={0: loc_link})

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        (mu,) = self.theta_to_params(theta)

        match param:
            case 0:
                return (y - mu) / mu

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        (mu,) = self.theta_to_params(theta)
        match param:
            case 0:
                # MU
                return -1 / mu

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        # No cross derivatives for single parameter distribution
        return np.zeros_like(y)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        initial_params = [np.mean(y)]
        return np.tile(initial_params, (y.shape[0], 1))

    def pmf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the probability mass function (PMF) for the given data points.

        Args:
            y (np.ndarray): An array of data points at which to evaluate the PMF.
            theta (np.ndarray): An array of parameters for the distribution.

        Returns:
            np.ndarray: An array of PMF values corresponding to the data points in `y`.
        """
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).pmf(y)

    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        For discrete distributions, PDF delegates to PMF.

        Args:
            y (np.ndarray): An array of data points.
            theta (np.ndarray): An array of parameters for the distribution.

        Returns:
            np.ndarray: An array of PMF values (same as pmf method).
        """
        return self.pmf(y, theta)

    def logpdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        For discrete distributions, log PDF delegates to log PMF.

        Args:
            y (np.ndarray): An array of data points.
            theta (np.ndarray): An array of parameters for the distribution.

        Returns:
            np.ndarray: An array of log PMF values (same as logpmf method).
        """
        return self.logpmf(y, theta)
