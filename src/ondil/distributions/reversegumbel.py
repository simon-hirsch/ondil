from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Identity, Log
from ..types import ParameterShapes


class ReverseGumbel(ScipyMixin, Distribution):
    r"""
    The Reverse Gumbel (Type I minimum extreme value) distribution with location (mu) and scale (sigma) parameters.

    The probability density function is defined as:
    $$
    f(y | \\mu, \\sigma) = \\frac{1}{\\sigma} \\exp\\left( \\frac{y - \\mu}{\\sigma} - \\exp\\left( \\frac{y - \\mu}{\\sigma} \\right) \\right)
    $$

    This distribution corresponds to the RG() distribution in GAMLSS.

    Notes:
        - Mean = mu - digamma(1) * sigma ≈ mu - 0.5772157 * sigma
        - Variance = (pi^2 * sigma^2) / 6 ≈ 1.64493 * sigma^2
    """

    corresponding_gamlss: str = "RG"
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
    scipy_dist = st.gumbel_r
    scipy_names = {"mu": "loc", "sigma": "scale"}

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
            }
        )

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        mu = theta[:, 0]
        sigma = theta[:, 1]
        return {
            "loc": mu,
            "scale": sigma,
        }

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)
        z = (y - mu) / sigma
        ez = np.exp(-z)

        if param == 0:
            return (1 - ez) / sigma

        if param == 1:
            return -(1 / sigma) * (1 - z) - (z / sigma) * ez

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        _, sigma = self.theta_to_params(theta)

        if param == 0:
            return -1 / sigma**2

        if param == 1:
            return -1.82368 / sigma**2  # approximation from GAMLSS

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        _, sigma = self.theta_to_params(theta)
        return np.full_like(y, -0.422784 / sigma**2)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        gamma_const = 0.5772157
        sigma_init = (np.sqrt(6) * np.std(y)) / np.pi
        mu_init = np.mean(y) + gamma_const * sigma_init
        initial_params = [mu_init, sigma_init]
        return np.tile(initial_params, (y.shape[0], 1))
