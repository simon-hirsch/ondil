from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Identity, Log
from ..types import ParameterShapes


class SkewNormal(ScipyMixin, Distribution):
    """
    The Skew Normal distribution with location, scale, and skewness parameterization.

    The probability density function is:
    $$
    f(y | \\mu, \\sigma, \\nu) = \\frac{2}{\\sigma} \\phi\\left(\\frac{y - \\mu}{\\sigma}\\right) \\Phi\\left(\\nu \\frac{y - \\mu}{\\sigma}\\right)
    $$

    where $\\phi$ is the standard normal PDF and $\\Phi$ is the standard normal CDF.

    This distribution corresponds to the SN1() distribution in GAMLSS.
    """

    corresponding_gamlss: str = "SN1"
    parameter_names = {0: "mu", 1: "sigma", 2: "nu"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (-np.inf, np.inf),
    }
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
        2: ParameterShapes.SCALAR,
    }
    distribution_support = (-np.inf, np.inf)
    scipy_dist = st.skewnorm
    scipy_names = {"mu": "loc", "sigma": "scale", "nu": "a"}

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
        skew_link: LinkFunction = Identity(),
    ) -> None:
        """Initialize the SkewNormal distribution.

        Args:
            loc_link (LinkFunction, optional): Location link. Defaults to Identity().
            scale_link (LinkFunction, optional): Scale link. Defaults to Log().
            skew_link (LinkFunction, optional): Skewness link. Defaults to Identity().
        """
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
                2: skew_link,
            }
        )

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        """Map GAMLSS parameters to scipy parameters.

        Args:
            theta (np.ndarray): parameters (mu, sigma, nu)

        Returns:
            dict: Dict of (a, loc, scale) for scipy.stats.skewnorm
        """
        mu = theta[:, 0]
        sigma = theta[:, 1]
        nu = theta[:, 2]
        return {"a": nu, "loc": mu, "scale": sigma}

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu = self.theta_to_params(theta)

        z = (y - mu) / sigma
        nuz = nu * z

        # Calculate phi(nu*z) / Phi(nu*z) ratio
        phi_nuz = st.norm.pdf(nuz)
        Phi_nuz = st.norm.cdf(nuz)
        # Avoid division by zero
        ratio = np.where(Phi_nuz > 1e-300, phi_nuz / Phi_nuz, 0.0)

        if param == 0:
            # dL/dmu
            return z / sigma + (nu / sigma) * ratio

        if param == 1:
            # dL/dsigma
            return -1 / sigma + (z**2) / sigma - (nu * z / sigma) * ratio

        if param == 2:
            # dL/dnu
            return z * ratio

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu = self.theta_to_params(theta)

        z = (y - mu) / sigma
        nuz = nu * z

        # Calculate phi(nu*z) / Phi(nu*z) ratio and derivative
        phi_nuz = st.norm.pdf(nuz)
        Phi_nuz = st.norm.cdf(nuz)
        ratio = np.where(Phi_nuz > 1e-300, phi_nuz / Phi_nuz, 0.0)

        if param == 0:
            # d2L/dmu2
            dl_dmu = z / sigma + (nu / sigma) * ratio
            return -(dl_dmu**2)

        if param == 1:
            # d2L/dsigma2
            dl_dsigma = -1 / sigma + (z**2) / sigma - (nu * z / sigma) * ratio
            return -(dl_dsigma**2)

        if param == 2:
            # d2L/dnu2
            dl_dnu = z * ratio
            return -(dl_dnu**2)

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        mu, sigma, nu = self.theta_to_params(theta)

        z = (y - mu) / sigma
        nuz = nu * z

        # Calculate phi(nu*z) / Phi(nu*z) ratio
        phi_nuz = st.norm.pdf(nuz)
        Phi_nuz = st.norm.cdf(nuz)
        ratio = np.where(Phi_nuz > 1e-300, phi_nuz / Phi_nuz, 0.0)

        if sorted(params) == [0, 1]:
            # d2L/dmu dsigma
            dl_dmu = z / sigma + (nu / sigma) * ratio
            dl_dsigma = -1 / sigma + (z**2) / sigma - (nu * z / sigma) * ratio
            return -(dl_dmu * dl_dsigma)

        if sorted(params) == [0, 2]:
            # d2L/dmu dnu
            dl_dmu = z / sigma + (nu / sigma) * ratio
            dl_dnu = z * ratio
            return -(dl_dmu * dl_dnu)

        if sorted(params) == [1, 2]:
            # d2L/dsigma dnu
            dl_dsigma = -1 / sigma + (z**2) / sigma - (nu * z / sigma) * ratio
            dl_dnu = z * ratio
            return -(dl_dsigma * dl_dnu)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        """Calculate initial values for the GAMLSS fit.

        For skew normal, we start with:
        - mu = mean of y
        - sigma = std of y
        - nu = 0 (no skewness initially)
        """
        initial_params = [np.mean(y), np.std(y, ddof=1), 0.0]
        return np.tile(initial_params, (y.shape[0], 1))
