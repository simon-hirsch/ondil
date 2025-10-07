# Author Simon Hirsch
# MIT Licence
from typing import Dict

import numpy as np
import scipy.stats as st


from ..base import Distribution, LinkFunction, CopulaMixin
from ..links import  FisherZLink, KendallsTauToParameter
from ..types import ParameterShapes


class BivariateCopulaNormal(CopulaMixin, Distribution):

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
        rotation: int = 0,
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
            rotation=rotation,

        )
        self.is_multivariate = True
        self._adr_lower_diag = {0: False}
        self._regularization_allowed = {0: False}
        self._regularization = ""  # or adr
        self._scoring = "fisher"


    @staticmethod
    def fitted_elements(dim: int):
        return {0: int(dim * (dim - 1) // 2)} 
    
    @property
    def param_structure(self):
        return self._param_structure


    @staticmethod
    def set_theta_element(theta: Dict, value: np.ndarray, param: int, k: int) -> Dict:
        """Sets an element of theta for parameter param and place k.

        !!! Note
            This will mutate `theta`!

        Args:
            theta (Dict): Current fitted $\theta$
            value (np.ndarray): Value to set
            param (int): Distribution parameter
            k (int): Flat element index $k$

        Returns:
            Dict: Theta where element (param, k) is set to value.
        """
        theta[param] = value
        return theta
    

    
    def theta_to_params(self, theta):
        if len(theta) > 1:
            chol = theta
        else:
            chol = theta[0]

        return chol
    
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

    def set_initial_guess(self, theta, param):
        return theta

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        """Return the first derivatives wrt to the parameter.

        !!! Note
            We expect the fitted L^-1)^T to be handed in matrix/cube form, i.e of shape n x d x d.
            But we return the derivatives in flat format.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with {0 : fitted mu, 1 : fitted (L^-1)^T}
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 1st derivatives.
        """
        chol = self.theta_to_params(theta)

        deriv = _derivative_1st(
            y=y, chol=chol
            )

        return deriv

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, clip=False):
        """Return the second derivatives wrt to the parameter.

        !!! Note
            We expect the fitted L^-1)^T to be handed in matrix/cube form, i.e of shape n x d x d.
            But we return the derivatives in flat format.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with {0 : fitted mu, 1 : fitted (L^-1)^T}
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 2nd derivatives.
        """
        fitted_loc = self.theta_to_params(theta)

        deriv = _derivative_2nd(
                y=y, fitted_loc=fitted_loc
        )
        return deriv

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False):
        chol = self.theta_to_params(theta)
        
        deriv = _derivative_1st(
                    y, chol
                )
        return deriv

    def element_dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False):
        fitted_loc = self.theta_to_params(theta)
              
        deriv = _derivative_2nd(
                    y, fitted_loc
                )
        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")


    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        
        return self.links[param].element_link(y)

    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        
        return self.links[param].element_derivative(y)

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
      
        return self.links[param].element_link_second_derivative(y)

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
     
        return self.links[param].inverse(y)

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
        ) -> np.ndarray:
      
        return self.links[param].element_inverse_derivative(y)

    def initial_values(self, y, param=0):
        M = y.shape[0]
        # Compute the empirical Pearson correlation for each sample
        # y is expected to be (M, 2)
        tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
        chol = np.full((M, 1), tau)
        return chol
    
    def cube_to_flat(self, x: np.ndarray, param: int):
        out = x
        return out

    def flat_to_cube(self, x: np.ndarray, param: int):
        out = x
        return out

    def param_conditional_likelihood(
        self, y: np.ndarray, theta: Dict, eta: np.ndarray, param: int
    ) -> np.ndarray:
        """Calulate the log-likelihood for (flat) eta for parameter (param)
        and theta for all other parameters.

        Args:
            y (np.ndarray): True values
            theta (Dict): Fitted theta.
            eta (np.ndarray): Fitted eta.
            param (int): Param for which we take eta.

        Returns:
            np.ndarray: Log-likelihood.
        """
        fitted = self.flat_to_cube(eta, param=param)
        fitted = self.link_inverse(fitted, param=param)
        # fitted_theta = {**theta, param: fitted_eta}
        return self.log_likelihood(y, theta={**theta, param: fitted})

    def theta_to_scipy(self, theta: Dict[int, np.ndarray]):
        out = {
            "cor": theta,
        }
        return out


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
        y = theta * z1 + np.sqrt(1 - theta ** 2) * z2

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

    def hfunc(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int) -> np.ndarray:
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
        M = u.shape[0]
        UMIN = 1e-12
        UMAX = 1 - 1e-12

        u = np.clip(u, UMIN, UMAX)
        v = np.clip(v, UMIN, UMAX)

        # Swap u and v if un == 1
        if un == 1:
            u, v = v, u

        # Handle edge cases
        h = np.where((v == 0) | (u == 0), 0, np.nan)
        h = np.where(v == 1, u, h)
        qnorm_u = st.norm.ppf(u)
        qnorm_v = st.norm.ppf(v)

        for m in range(M):
            denom = np.sqrt(1.0 - np.square(theta[m]))
            x = (qnorm_u[m] - theta[m] * qnorm_v[m]) / denom
            if np.isfinite(x):
                h[m] = st.norm.cdf(x)
            elif (qnorm_u[m] - theta[m] * qnorm_v[m]) < 0:
                h[m] = 0
            else:
                h[m] = 1

        # Clip output for numerical stability
        h = np.clip(h, UMIN, UMAX)
        return h

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
        M = u.shape[0]
        UMIN = 1e-12
        UMAX = 1 - 1e-12

        u = np.clip(u, UMIN, UMAX)
        v = np.clip(v, UMIN, UMAX)

        # Swap u and v if un == 1
        if un == 1:
            u, v = v, u

        hinv = np.empty(M)
        qnorm_u = st.norm.ppf(u)
        qnorm_v = st.norm.ppf(v)

        for m in range(M):
            theta_m = theta[0][m] if hasattr(theta[0], '__len__') else theta[0]
            x = qnorm_u[m] * np.sqrt(1.0 - theta_m**2) + theta_m * qnorm_v[m]
            hinv[m] = st.norm.cdf(x)

        # Clip output for numerical stability
        hinv = np.clip(hinv, UMIN, UMAX)
        return hinv


    def get_regularization_size(self, dim: int) -> int:
        return dim
##########################################################
### numba JIT-compiled functions for the derivatives #####
##########################################################


def _log_likelihood(y, mod_chol):
    M = y.shape[0]
    f = np.empty(M)
    # Ensure y values are strictly between 0 and 1 for numerical stability
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    y_clipped = np.clip(y, UMIN, UMAX)
    u = st.norm().ppf(y_clipped[:, 0])
    v = st.norm().ppf(y_clipped[:, 1])
    for m in range(M):
        if M == 1:
            rho = mod_chol
        else: 
            rho = mod_chol[m]
        t1 = u[m]
        t2 = v[m]
        f[m] = float(
            1.0 / np.sqrt(1.0 - rho**2)
            * np.exp(
                (t1**2 + t2**2) / 2.0
                + (2.0 * rho * t1 * t2 - t1**2 - t2**2) / (2.0 * (1.0 - rho**2))
            )
        )
    # Replace any zeros in f with 1e-16 for numerical stability
    f[f == 0] = 1e-16
    return f

def _derivative_1st(y, chol):
    """
    Implements the first derivative of the bivariate Gaussian copula log-likelihood
    with respect to the correlation parameter, following the C++ code logic.

    Args:
        y (np.ndarray): Input data of shape (M, 2)
        chol (np.ndarray): Correlation parameter, shape (M,) or (1, M)

    Returns:
        np.ndarray: First derivative, shape (M,)
    """
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    eps = np.finfo(float).eps
    y = np.clip(y, eps, 1 - eps)
    u = st.norm().ppf(y[:, 0])
    v = st.norm().ppf(y[:, 1])
    for m in range(M):
        if M == 1:
            theta  = chol
        else:
            theta = chol[m]
        t3 = theta*theta
        t4 = 1.0-t3
        t5 =  u[m]*u[m]
        t6 = v[m]*v[m]
        t7 = t4*t4
        deriv[m] =  (theta*t4-theta*(t5+t6)+(1.0+t3)*u[m]*v[m])/t7

    return deriv.squeeze()

def _derivative_2nd(y, fitted_loc): 
    
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)

    # Ensure y values are strictly between 0 and 1 for numerical stability
    eps = np.finfo(float).eps
    y = np.clip(y, eps, 1 - eps)
    u = st.norm().ppf(y[:, 0])
    v = st.norm().ppf(y[:, 1])

    for m in range(M):
        if M == 1:
            theta  = fitted_loc
        else:
            theta = fitted_loc[m]
        t6 = u[m]
        t7 = v[m]
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
        deriv[m] = (
            (16.0 * t1 * t5 * theta + 16.0 * t14 / t4 / t3 * t2 + 4.0 * t21) * t29 * t32
            + t39 * t29 * t32
            + 2.0 * t38 * t29 * t44 * theta
            + 3.0 * t29 / t31 / t48 * t2
            + t29 * t44
        )
    return deriv.squeeze()


	