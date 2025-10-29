# Author Simon Hirsch
# MIT Licence
from typing import Dict

import numpy as np
import scipy.stats as st

from ..base import BivariateCopulaMixin, CopulaMixin, Distribution, LinkFunction
from ..links import FisherZLink, KendallsTauToParameter
from ..types import ParameterShapes


class BivariateCopulaNormal(BivariateCopulaMixin, CopulaMixin, Distribution):

    corresponding_gamlss: str = None
    parameter_names = {0: "rho"}
    parameter_support = {0: (-1, 1)}
    distribution_support = (-1, 1)
    n_params = len(parameter_names)
    parameter_shape = {
        0: ParameterShapes.SCALAR,
    }

    def __init__(
        self,
        link: LinkFunction = FisherZLink(),
        param_link: LinkFunction = KendallsTauToParameter(),
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
        )
        self.is_multivariate = True
        self._regularization_allowed = {0: False}

    @staticmethod
    def fitted_elements(dim: int):
        return {0: 1}

    def theta_to_params(self, theta) -> np.ndarray:
        return theta[0]

    def set_initial_guess(self, theta, param):
        return theta

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        """Return the first derivatives wrt to the parameter.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with {0 : fitted mu, 1 : fitted (L^-1)^T}
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 1st derivatives.
        """
        rho = self.theta_to_params(theta)
        deriv = _derivative_1st(y=y, rho=rho)
        return deriv

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, clip=False):
        """Return the second derivatives wrt to the parameter.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with {0 : fitted mu, 1 : fitted (L^-1)^T}
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 2nd derivatives.
        """
        fitted_loc = self.theta_to_params(theta)
        deriv = _derivative_2nd(y=y, fitted_loc=fitted_loc)
        return deriv

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        rho = self.theta_to_params(theta)
        deriv = _derivative_1st(y, rho)
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        fitted_loc = self.theta_to_params(theta)
        deriv = _derivative_2nd(y, fitted_loc)
        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def initial_values(self, y, param=0):
        M = y.shape[0]
        # Compute the empirical Pearson correlation for each sample
        # y is expected to be (M, 2)
        tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
        rho = np.full((M, 1), tau)
        return rho

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        """
        Generate random samples from the bivariate normal copula.

        Args:
            size (int): Number of samples to generate.
            theta (dict or np.ndarray): Correlation parameter(s).

        Returns:
            np.ndarray: Samples of shape (size, 2) in (0, 1).
        """

        # Generate standard normal samples
        z1 = np.random.normal(size=size)
        z2 = np.random.normal(size=size)
        x = z1
        y = theta * z1 + np.sqrt(1 - theta**2) * z2

        # Transform to uniform marginals using the normal CDF
        u = st.norm.cdf(x)
        v = st.norm.cdf(y)
        return np.column_stack((u, v))

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpdf(self, y, theta):
        test = self.theta_to_params(theta)
        return np.log(_log_likelihood(y, test))

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")

    def get_regularization_size(self, dim: int) -> int:
        return dim

    def hfunc(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int
    ) -> np.ndarray:
        """
        Conditional distribution function h(u|v) for the bivariate normal copula.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (np.ndarray or float): Correlation parameter(s), shape (n,) or scalar.
            un (int): Determines which conditional to compute (0 for h(u|v), 1 for h(v|u)).

        Returns:
            np.ndarray: Array of shape (n,) with conditional probabilities.
        """
        UMIN = 1e-12
        UMAX = 1 - 1e-12

        # Apply clipping using masks
        u_mask_low = u < UMIN
        u_mask_high = u > UMAX
        v_mask_low = v < UMIN
        v_mask_high = v > UMAX
        
        u = np.where(u_mask_low, UMIN, u)
        u = np.where(u_mask_high, UMAX, u)
        v = np.where(v_mask_low, UMIN, v)
        v = np.where(v_mask_high, UMAX, v)

        qnorm_u = st.norm.ppf(u).reshape(-1, 1)
        qnorm_v = st.norm.ppf(v).reshape(-1, 1)

        denom = np.sqrt(1.0 - theta**2)
        x = (qnorm_u - theta * qnorm_v) / denom

        h = np.where(np.isfinite(x), st.norm.cdf(x), 
                np.where((qnorm_u - theta * qnorm_v) < 0, 0, 1))

        # Ensure results are in [0,1] using masks
        h_mask_low = h < 0
        h_mask_high = h > 1
        h = np.where(h_mask_low, 0, h)
        h = np.where(h_mask_high, 1, h)

        return h.squeeze()
    

    def hinv(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int) -> np.ndarray:
        """
        Inverse conditional distribution function h^(-1)(u|v) for the bivariate normal copula.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (np.ndarray or float): Correlation parameter(s), shape (n,) or scalar.
            un (int): Determines which conditional to compute.

        Returns:
            np.ndarray: Array of shape (n,) with inverse conditional probabilities.
        """

        UMIN = 1e-12
        UMAX = 1 - 1e-12

        # Apply clipping using masks
        u_mask_low = u < UMIN
        u_mask_high = u > UMAX
        v_mask_low = v < UMIN
        v_mask_high = v > UMAX
        
        u = np.where(u_mask_low, UMIN, u)
        u = np.where(u_mask_high, UMAX, u)
        v = np.where(v_mask_low, UMIN, v)
        v = np.where(v_mask_high, UMAX, v)
        
        qnorm_u = st.norm.ppf(u).reshape(-1, 1)
        qnorm_v = st.norm.ppf(v).reshape(-1, 1)

        x = qnorm_u * np.sqrt(1.0 - theta**2) + theta * qnorm_v
        hinv = st.norm.cdf(x)

        # Clip output for numerical stability
        # Ensure results are in [0,1] using masks

        h_mask_low = hinv < 0
        h_mask_high = hinv > 1
        hinv = np.where(h_mask_low, 0, hinv)
        hinv = np.where(h_mask_high, 1, hinv)

        return hinv.squeeze()



##########################################################
### numba JIT-compiled functions for the derivatives #####
##########################################################


def _log_likelihood(y, theta):

    M = y.shape[0]

    f = np.empty(M)
    # Ensure y values are strictly between 0 and 1 for numerical stability
    UMIN = 1e-12
    UMAX = 1 - 1e-12

    y_clipped = np.clip(y, UMIN, UMAX)

    u = st.norm().ppf(y_clipped[:, 0]).reshape(-1, 1)
    v = st.norm().ppf(y_clipped[:, 1]).reshape(-1, 1)

    t1 = u
    t2 = v
    f = (
        1.0
        / np.sqrt(1.0 - theta**2)
        * np.exp(
            (t1**2 + t2**2) / 2.0
            + (2.0 * theta * t1 * t2 - t1**2 - t2**2) / (2.0 * (1.0 - theta**2))
        )
    )
    # Replace any zeros in f with 1e-16 for numerical stability
    f[f == 0] = 1e-16

    return f.squeeze()


def _derivative_1st(y, theta):
    """
    Implements the first derivative of the bivariate Gaussian copula log-likelihood
    with respect to the correlation parameter

    Args:
        y (np.ndarray): Input data of shape (M, 2)
        rho (np.ndarray): Correlation parameter, shape (M,) or (1, M)

    Returns:
        np.ndarray: First derivative, shape (M,)
    """

    UMIN = 1e-12
    UMAX = 1 - 1e-12
    y_clipped = np.clip(y, UMIN, UMAX)
    u = st.norm().ppf(y_clipped[:, 0]).reshape(-1, 1)
    v = st.norm().ppf(y_clipped[:, 1]).reshape(-1, 1)

    t3 = theta * theta
    t4 = 1.0 - t3
    t5 = u * u
    t6 = v * v
    t7 = t4 * t4
    deriv = (theta * t4 - theta * (t5 + t6) + (1.0 + t3) * u * v) / t7
    return deriv.squeeze()


def _derivative_2nd(y, theta):

    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)

    # Ensure y values are strictly between 0 and 1 for numerical stability
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    y_clipped = np.clip(y, UMIN, UMAX)
    u = st.norm().ppf(y_clipped[:, 0]).reshape(-1, 1)
    v = st.norm().ppf(y_clipped[:, 1]).reshape(-1, 1)

    t6 = u
    t7 = v
    t1 = t6 * t7
    t2 = theta * theta
    t3 = 1.0 - t2
    t4 = 4.0 * t3 * t3
    t5 = 1.0 / t4
    t12 = t6 * t6
    t13 = t7 * t7
    t14 = 2.0 * theta * t6 * t7 - t12 - t13
    t21 = t14 * t5
    t26 = 1.0 / t3 / 2.0
    t29 = np.exp(t12 / 2.0 + t13 / 2.0 + t14 * t26)
    t31 = np.sqrt(t3)
    t32 = 1.0 / t31
    t38 = 2.0 * t1 * t26 + 4.0 * t21 * theta
    t39 = t38 * t38
    t44 = 1.0 / t31 / t3
    t48 = t3 * t3
    deriv = (
        (16.0 * t1 * t5 * theta + 16.0 * t14 / t4 / t3 * t2 + 4.0 * t21) * t29 * t32
        + t39 * t29 * t32
        + 2.0 * t38 * t29 * t44 * theta
        + 3.0 * t29 / t31 / t48 * t2
        + t29 * t44
    )
    return deriv.squeeze()
