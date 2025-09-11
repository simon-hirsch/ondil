from typing import Dict, Optional, Tuple

import numpy as np
import scipy.stats as st
from scipy.special import digamma, polygamma

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Log


class InverseGamma(ScipyMixin, Distribution):
    """
    The Inverse Gamma distribution as parameterized in GAMLSS:

    The distribution has two parameters:
        - mu: mean-related parameter
        - sigma: dispersion parameter

    Reparameterization:
        α = 1 / sigma²
        scale = mu * (1 + sigma²) / sigma²

    This distribution corresponds to IGAMMA() in GAMLSS.
    """

    corresponding_gamlss: str = "IGAMMA"
    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (np.nextafter(0, 1), np.inf),
        1: (np.nextafter(0, 1), np.inf),
    }
    distribution_support = (0, np.inf)
    scipy_dist = st.invgamma
    scipy_names = {"mu": "scale", "sigma": "a"}

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

    def theta_to_scipy_params(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        mu, sigma = self.theta_to_params(theta)
        alpha = 1.0 / sigma**2
        scale = mu * (1 + sigma**2) / sigma**2
        return {"a": alpha, "scale": scale, "loc": 0}

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)
        alpha = 1.0 / sigma**2

        if param == 0:
            return (alpha / mu) - ((alpha + 1) / y)
        if param == 1:
            term = (
                np.log(mu)
                + (alpha / (alpha + 1))
                + np.log(alpha + 1)
                - digamma(alpha)
                - np.log(y)
                - (mu / y)
            )
            return (-2 / sigma**3) * term

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)
        alpha = 1.0 / sigma**2

        if param == 0:
            return -1.0 / (sigma**2 * mu**2)
        elif param == 1:
            part1 = (sigma**2 * (1 + 2 * sigma**2)) / ((1 + sigma**2) ** 2)
            part2 = polygamma(1, alpha)
            return -4 * (part2 - part1) / (sigma**6)
        else:
            raise ValueError(f"Invalid parameter index: {param}")

    def sample(self, n: int, theta: np.ndarray) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        alpha = 1.0 / sigma**2
        scale = mu * (1 + sigma**2) / sigma**2
        return st.invgamma.rvs(a=alpha, scale=scale, size=n)

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        mu, sigma = self.theta_to_params(theta)
        if sorted(params) == [0, 1]:
            return -(2 / (mu * sigma * (1 + sigma**2)))

    def initial_values(
        self, y: np.ndarray, param: Optional[int] = None, axis: Optional[int] = None
    ) -> np.ndarray:
        y = np.asarray(y)
        mu_init = np.mean(y, axis=axis)
        sigma_init = np.std(y, ddof=1) / np.sqrt((mu_init))
        if param == 0:
            return np.full_like(y, mu_init)
        elif param == 1:
            return np.full_like(y, sigma_init)
        initial_params = [mu_init, sigma_init]
        return np.tile(initial_params, (y.shape[0], 1))
