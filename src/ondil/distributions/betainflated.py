from typing import Tuple

import numpy as np
import scipy.special as spc
import scipy.stats as st

from ..base import Distribution, LinkFunction
from ..links import Log, Logit
from ..types import ParameterShapes


class BetaInflated(Distribution):
    r"""The Beta Inflated Distribution for GAMLSS.
    
    The distribution function is defined as in GAMLSS as:
    $$
    f_Y(y \\mid \\mu, \\sigma, \\nu, \\tau) = 
    \\begin{cases}
    p_0 & \\text{if } y = 0 \\
    (1 - p_0 - p_1) \\dfrac{1}{B(\\alpha, \\beta)} y^{\\alpha - 1}(1 - y)^{\\beta - 1} & \\text{if } 0 < y < 1 \\
    p_1 & \\text{if } y = 1
    \\end{cases}
    $$
       
    where $\\alpha = \\mu (1 - \\sigma^2) / \\sigma^2$, \\beta = (1 - \\mu) (1 - \\sigma^2)/ \\sigma^2; 
    p_0 = \\nu (1 + \\nu + \\tau)^{-1} and p_1 =  \\tau (1 + \\nu + \\tau)^{-1}$, 

    and $\\mu, \\sigma \\in (0,1)$ and $\\nu, \\tau > 0 $

    The parameter tuple $\\theta$ in Python is defined as:

    $\\theta = (\\theta_0, \\theta_1, \\theta_2, \\theta_3) = (\\mu, \\sigma, \\nu, \\tau)$ 
    where $\\mu = \\theta_0$ is the location parameter, $\\sigma = \\theta_1$ is the scale parameter 
    and $\\nu, \\tau = \\theta_2, \\theta_3$ are shape parameters which together define the inflation at 0 and 1

    This distribution corresponds to the BEINF() distribution in GAMLSS.
    """

    corresponding_gamlss: str = "BEINF"

    parameter_names = {0: "mu", 1: "sigma", 2: "nu", 3: "tau"}
    parameter_support = {
        0: (np.nextafter(0, 1), np.nextafter(1, 0)),
        1: (np.nextafter(0, 1), np.nextafter(1, 0)),
        2: (np.nextafter(0, 1), np.inf),  ##
        3: (np.nextafter(0, 1), np.inf),  ##
    }
    distribution_support = (0, 1)
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
        2: ParameterShapes.SCALAR,
        3: ParameterShapes.SCALAR,
    }

    def __init__(
        self,
        loc_link: LinkFunction = Logit(),
        scale_link: LinkFunction = Logit(),
        skew_link: LinkFunction = Log(),  ##
        tail_link: LinkFunction = Log(),  ##
    ) -> None:
        super().__init__(links={0: loc_link, 1: scale_link, 2: skew_link, 3: tail_link})

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu, tau = self.theta_to_params(theta)

        if param == 0:
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            result = ((1 - sigma**2) / sigma**2) * (
                -spc.digamma(alpha) + spc.digamma(beta) + np.log(y) - np.log(1 - y)
            )

            return np.where((y == 0) | (y == 1), 0, result)

        if param == 1:
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            result = -(2 / sigma**3) * (
                mu * (-spc.digamma(alpha) + spc.digamma(alpha + beta) + np.log(y))
                + (1 - mu)
                * (-spc.digamma(beta) + spc.digamma(alpha + beta) + np.log(1 - y))
            )

            return np.where((y == 0) | (y == 1), 0, result)

        if param == 2:
            return np.where(
                y == 0, (1 / nu) - (1 / (1 + nu + tau)), 0 - (1 / (1 + nu + tau))
            )

        if param == 3:
            return np.where(
                y == 1, (1 / tau) - (1 / (1 + nu + tau)), 0 - (1 / (1 + nu + tau))
            )

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu, tau = self.theta_to_params(theta)
        if param == 0:
            # MU
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            result = -(((1 - sigma**2) ** 2) / sigma**4) * (
                spc.polygamma(1, alpha) + spc.polygamma(1, beta)
            )

            return np.where((y == 0) | (y == 1), 0, result)

        if param == 1:
            # SIGMA
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            result = -(4 / (sigma**6)) * (
                (mu**2) * spc.polygamma(1, alpha)
                + ((1 - mu) ** 2) * spc.polygamma(1, beta)
                - spc.polygamma(1, alpha + beta)
            )

            return np.where((y == 0) | (y == 1), 0, result)

        if param == 2:
            return -(1 + tau) / (nu * ((1 + nu + tau) ** 2))

        if param == 3:
            return -(1 + nu) / (tau * ((1 + nu + tau) ** 2))

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        mu, sigma, nu, tau = self.theta_to_params(theta)

        if sorted(params) == [0, 1]:
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            result = (2 * (1 - sigma**2) / sigma**5) * (
                mu * spc.polygamma(1, alpha) - (1 - mu) * spc.polygamma(1, beta)
            )

            return np.where((y == 0) | (y == 1), 0, result)

        if sorted(params) == [0, 2]:
            return np.zeros_like(y)  ###

        if sorted(params) == [0, 3]:
            return np.zeros_like(y)  ###

        if sorted(params) == [1, 2]:
            return np.zeros_like(y)  ###

        if sorted(params) == [1, 3]:
            return np.zeros_like(y)  ###

        if sorted(params) == [2, 3]:
            return 1 / (1 + nu + tau) ** 2  ###

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        return np.tile([np.mean(y), 0.5, 5, 5], (y.shape[0], 1))

    def cdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        alpha = mu * (1 - sigma**2) / sigma**2
        beta = alpha * (1 - mu) / mu

        cdf_beta = st.beta(alpha, beta).cdf(y)

        raw_cdf = np.where(
            (y > 0) & (y < 1),
            nu + cdf_beta,
            np.where(y == 0, nu, np.where(y == 1, 1 + nu + tau, 0)),
        )

        result = raw_cdf / (1 + nu + tau)

        result = np.where(y < 0, 0, result)
        result = np.where(y > 1, 1, result)

        return result

    def pdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        alpha = mu * (1 - sigma**2) / sigma**2
        beta = (1 - mu) * (1 - sigma**2) / sigma**2

        pdf_beta = st.beta(alpha, beta, loc=0, scale=1).pdf(y)

        result = np.where(
            (y < 0) | (y > 1),
            0,
            np.where(
                y == 0,
                nu / (1 + nu + tau),
                np.where(y == 1, tau / (1 + nu + tau), pdf_beta / (1 + nu + tau)),
            ),
        )
        return result

    def ppf(self, q, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)

        alpha = mu * (1 - sigma**2) / sigma**2
        beta = (1 - mu) * (1 - sigma**2) / sigma**2

        denom = 1 + nu + tau
        lower = nu / denom
        upper = (1 + nu) / denom

        q = np.asarray(q, dtype=np.float64)
        adjusted_q = (q - lower) * denom  # equivalent to (q - lower) / (1/denom)

        finite_result = st.beta.ppf(adjusted_q, a=alpha, b=beta)

        one_result = np.where(np.logical_or(q == 1, q >= upper), 1.0, finite_result)
        zero_result = np.where(np.logical_or(q == 0, q <= lower), 0.0, one_result)
        result = np.where(np.logical_or(q < 0, q > 1), np.nan, zero_result)

        return result

    def pmf(self, y, theta):
        raise NotImplementedError("PMF is not implemented for mixed distributions.")

    def rvs(self, size, theta):
        sim_unif = st.uniform.rvs(size=size)
        return self.ppf(q=sim_unif, theta=theta)

    def logpdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        alpha = mu * (1 - sigma**2) / sigma**2
        beta = (1 - mu) * (1 - sigma**2) / sigma**2

        logpdf_beta = st.beta(alpha, beta, loc=0, scale=1).logpdf(y)

        result = np.where(
            (y < 0) | (y > 1),
            0,
            np.where(
                y == 0,
                np.log(nu) - np.log(1 + nu + tau),
                np.where(
                    y == 1,
                    np.log(tau) - np.log(1 + nu + tau),
                    logpdf_beta - np.log(1 + nu + tau),
                ),
            ),
        )
        return result

    def logcdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        alpha = mu * (1 - sigma**2) / sigma**2
        beta = (1 - mu) * (1 - sigma**2) / sigma**2

        y = np.asarray(y, dtype=np.float64)
        denom = 1 + nu + tau

        cdf_beta = st.beta(alpha, beta).cdf(y)

        raw_cdf = np.where(
            (y > 0) & (y < 1),
            nu + cdf_beta,
            np.where(y == 0, nu, np.where(y == 1, denom, 0)),
        )

        with np.errstate(divide="ignore"):
            log_result = np.log(raw_cdf / denom)

        log_result = np.where(y < 0, -np.inf, log_result)
        log_result = np.where(y > 1, 0.0, log_result)

        return log_result

    def logpmf(self, y, theta):
        return super().logpmf(y, theta)

    def calculate_conditional_initial_values(self, y, theta, param):
        return super().calculate_conditional_initial_values(y, theta, param)
