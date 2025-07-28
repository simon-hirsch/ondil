from typing import Tuple

import numpy as np
import scipy.special as spc
import scipy.stats as st

from ..base import Distribution, LinkFunction
from ..links import Logit, Log


class BetaInflatedZero(Distribution):
    """The Zero Inflated Beta Distribution for GAMLSS.
    
    f_Y(y \\mid \\mu, \\sigma, \\nu) = 
    \\begin{cases}
    p_0 & \text{if } y = 0 \\
    (1 - p_0) f_W(y \mid \mu, \sigma) & \\text{if } 0 < y < 1
    \\end{cases}
        
    where  $p_0 = \\nu (1 + \\nu)^{-1}$
    
    and $\\mu, \\sigma \\in (0,1)$ and $\\nu > 0 $
    
    """

    corresponding_gamlss: str = "BEINF0"

    parameter_names = {0: "mu", 1: "sigma", 2: "nu"}
    parameter_support = {
        0: (np.nextafter(0, 1), np.nextafter(1, 0)),
        1: (np.nextafter(0, 1), np.nextafter(1, 0)),
        2: (np.nextafter(0, 1), np.inf),
    }
    distribution_support = (0, np.nextafter(1, 0))

    def __init__(
        self,
        loc_link: LinkFunction = Logit(),
        scale_link: LinkFunction = Logit(),
        inflation_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(links={0: loc_link, 1: scale_link, 2: inflation_link})

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu = self.theta_to_params(theta)

        if param == 0:
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            result = ((1 - sigma**2) / sigma**2) * (
                -spc.digamma(alpha) + spc.digamma(beta) + np.log(y) - np.log(1 - y)
            )

            return np.where(y == 0, 0, result)

        if param == 1:
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            result = -(2 / sigma**3) * (
                mu * (-spc.digamma(alpha) + spc.digamma(alpha + beta) + np.log(y))
                + (1 - mu)
                * (-spc.digamma(beta) + spc.digamma(alpha + beta) + np.log(1 - y))
            )

            return np.where(y == 0, 0, result)

        if param == 2:
            return np.where(y == 0, (1 / nu), 0) - (1 / (1 + nu))

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu = self.theta_to_params(theta)
        if param == 0:
            # MU
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            result = -(((1 - sigma**2) ** 2) / sigma**4) * (
                spc.polygamma(1, alpha) + spc.polygamma(1, beta)
            )

            return np.where(y == 0, 0, result)

        if param == 1:
            # SIGMA
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            result = -(4 / (sigma**6)) * (
                (mu**2) * spc.polygamma(1, alpha)
                + ((1 - mu) ** 2) * spc.polygamma(1, beta)
                - spc.polygamma(1, alpha + beta)
            )

            return np.where(y == 0, 0, result)

        if param == 2:
            return -1 / (nu * ((1 + nu) ** 2))

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        mu, sigma, nu = self.theta_to_params(theta)

        if sorted(params) == [0, 1]:
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            result = (2 * (1 - sigma**2) / sigma**5) * (
                mu * spc.polygamma(1, alpha) - (1 - mu) * spc.polygamma(1, beta)
            )

            return np.where(y == 0, 0, result)

        if sorted(params) == [0, 2]:
            return np.zeros_like(y)  ###

        if sorted(params) == [1, 2]:
            return np.zeros_like(y)  ###

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        return np.tile([np.mean(y), 0.5, np.mean(y)], (y.shape[0], 1))

    def cdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        alpha = mu * (1 - sigma**2) / sigma**2
        beta = alpha * (1 - mu) / mu

        cdf_beta = st.beta(alpha, beta).cdf(y)

        raw_cdf = np.where(y == 0, nu, nu + cdf_beta)

        result = raw_cdf / (1 + nu)

        result = np.where(y < 0, 0, result)
        result = np.where(y >= 1, 1, result)

        return result

    def pdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        alpha = mu * (1 - sigma**2) / sigma**2
        beta = (1 - mu) * (1 - sigma**2) / sigma**2

        pdf_beta = st.beta(alpha, beta, loc=0, scale=1).pdf(y)

        result = np.where(
            (y < 0) | (y >= 1),
            0,
            np.where(y == 0, nu / (1 + nu), pdf_beta / (1 + nu)),
        )

        return result

    def ppf(self, q, theta):
        mu, sigma, nu = self.theta_to_params(theta)

        alpha = mu * (1 - sigma**2) / sigma**2
        beta = (1 - mu) * (1 - sigma**2) / sigma**2

        denom = 1 + nu
        lower = nu / denom

        q = np.asarray(q, dtype=np.float64)
        adjusted_q = (q - lower) * denom  # same as (q - lower) / (1/denom)

        finite_result = st.beta.ppf(adjusted_q, a=alpha, b=beta)

        finite_result = np.where(q <= lower, 0.0, finite_result)
        finite_result = np.where(q == 0, 0.0, finite_result)
        finite_result = np.where(q == 1, np.nextafter(1, 0), finite_result)
        result = np.where((q < 0) | (q > 1), np.nan, finite_result)

        return result

    def pmf(self, y, theta):
        raise NotImplementedError("PMF is not implemented for mixed distributions.")

    def rvs(self, size, theta):
        sim_unif = st.uniform.rvs(size=size)
        return self.ppf(q=sim_unif, theta=theta)

    def logpdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        alpha = mu * (1 - sigma**2) / sigma**2
        beta = (1 - mu) * (1 - sigma**2) / sigma**2

        logpdf_beta = st.beta(alpha, beta).logpdf(y)

        logfy = np.zeros_like(y, dtype=np.float64)
        logfy = np.where(y > 0, logpdf_beta, logfy)
        logfy = np.where(y == 0, np.log(nu), logfy)
        result = logfy - np.log(1 + nu)

        return result

    def logcdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        alpha = mu * (1 - sigma**2) / sigma**2
        beta = (1 - mu) * (1 - sigma**2) / sigma**2

        y = np.asarray(y, dtype=np.float64)
        denom = 1 + nu

        cdf_beta = st.beta(alpha, beta).cdf(y)

        raw_cdf = np.where((y > 0) & (y <= 1), nu + cdf_beta, np.where(y == 0, nu, 0))

        with np.errstate(divide="ignore"):
            log_result = np.log(raw_cdf / denom)

        log_result = np.where(y < 0, -np.inf, log_result)
        log_result = np.where(y >= 1, 0.0, log_result)

        return log_result

    def logpmf(self, y, theta):
        return super().logpmf(y, theta)

    def calculate_conditional_initial_values(self, y, theta, param):
        return super().calculate_conditional_initial_values(y, theta, param)
