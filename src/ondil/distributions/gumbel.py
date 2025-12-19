import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Identity, Log
from ..types import ParameterShapes


class Gumbel(ScipyMixin, Distribution):
    r"""
    The Gumbel distribution.

    The probability density function is given by:
    $$
        f(y|\\mu, \\sigma) = (1/\\sigma) * \\exp(-(z + \\exp(-z)))
    $$
    where $z = (y - \\mu)/\\sigma$ and has the following parameters:

    * $\\mu$: location
    * $\\sigma$: scale (>0)
    """

    corresponding_gamlss: str = "GU"
    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
    }
    distribution_support = (-np.inf, np.inf)
    scipy_dist = st.gumbel_l
    scipy_names = {"mu": "loc", "sigma": "scale"}
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(links={0: loc_link, 1: scale_link})

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        z = (y - mu) / sigma
        if param == 0:
            # d logL / d mu
            # (exp((y-mu)/sigma)-1)/sigma,
            return (np.exp(z) - 1) / sigma
        if param == 1:
            # d logL / d sigma#
            # -(1/sigma)+  ((y-mu)/sigma2)*(exp((y-mu)/sigma)-1),
            return -(1 / sigma) + (z / sigma) * (np.exp(z) - 1)
        raise ValueError("param must be 0 (mu) or 1 (sigma)")

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        _, sigma = self.theta_to_params(theta)
        if param == 0:
            # d^2 logL / d mu^2
            return -1 / sigma**2
        if param == 1:
            # d^2 logL / d sigma^2
            return -1.82368 / sigma**2
        raise ValueError("param must be 0 (mu) or 1 (sigma)")

    def dl2_dpp(self, y: np.ndarray, theta: np.ndarray, params=(0, 1)) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        _, sigma = self.theta_to_params(theta)
        if sorted(params) == [0, 1]:
            # d^2 logL / d mu d sigma
            # Note: order does not matter for cross-derivative
            return -0.422784 / sigma**2
        raise ValueError("Cross derivatives must use different parameters.")

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        out = np.zeros((y.shape[0], 2))
        out[:, 0] = np.mean(y, axis=0)
        out[:, 1] = np.sqrt(6) * np.std(y, axis=0) / np.pi
        return out
