# Author Simon Hirsch
# MIT Licence
from typing import Dict

import numpy as np
import scipy.stats as st


from ..base import Distribution, LinkFunction, CopulaMixin
from ..links import  GumbelLink, KendallsTauToParameter, KendallsTauToParameterGumbel
from ..types import ParameterShapes
from scipy.optimize import brentq


class BivariateCopulaGumbel(CopulaMixin, Distribution):

    corresponding_gamlss: str = None
    parameter_names = {0: "theta"}
    parameter_support = {0: (1, np.inf)}
    distribution_support = (0, 1) 
    n_params = len(parameter_names)
    parameter_shape = {
        0: ParameterShapes.SCALAR,
    }
    def __init__(
        self, 
        link: LinkFunction = GumbelLink(), 
        param_link: LinkFunction = KendallsTauToParameterGumbel(),
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
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
            theta_param = theta
        else:
            theta_param = theta[0]

        return theta_param
    
    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        """Map GAMLSS Parameters to scipy parameters.

        Args:
            theta (np.ndarray): parameters

        Returns:
            dict: Dict of parameters for Gumbel copula
        """
        theta_param = theta[:, 0]
        params = {"theta": theta_param}
        return params

    def set_initial_guess(self, theta, param):
        return theta

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        """Return the first derivatives wrt to the parameter.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with {0 : fitted theta}
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 1st derivatives.
        """
        theta_param = self.theta_to_params(theta)

        deriv = _derivative_1st(
            y=y, theta=theta_param
            )

        return deriv

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, clip=False):
        """Return the second derivatives wrt to the parameter.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with {0 : fitted theta}
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 2nd derivatives.
        """
        theta_param = self.theta_to_params(theta)

        deriv = _derivative_2nd(
                y=y, theta=theta_param
        )
        return deriv

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False):
        theta_param = self.theta_to_params(theta)
        
        deriv = _derivative_1st(
                    y, theta_param
                )
        return deriv

    def element_dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False):
        theta_param = self.theta_to_params(theta)
              
        deriv = _derivative_2nd(
                    y, theta_param
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
        # Compute the empirical Kendall's tau and convert to Gumbel parameter
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
        return self.log_likelihood(y, theta={**theta, param: fitted})

    def theta_to_scipy(self, theta: Dict[int, np.ndarray]):
        out = {
            "theta": theta,
        }
        return out


    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        """
        Generate random samples from the bivariate Gumbel copula.

        Args:
            size (int): Number of samples to generate.
            theta (dict or np.ndarray): Gumbel parameter(s).

        Returns:
            np.ndarray: Samples of shape (size, 2) in (0, 1).
        """
        # Use rejection sampling or other methods for Gumbel copula
        # This is a simplified implementation
        raise NotImplementedError("Gumbel copula sampling not implemented")


    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpdf(self, y, theta):
        theta_param = self.theta_to_params(theta)
        return _log_likelihood(y, theta_param)

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
        Conditional distribution function h(u|v) for the bivariate Gumbel copula.
        Implementation based on vinecopulib package.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (np.ndarray or float): Gumbel parameter(s), shape (n,) or scalar.
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

        h = np.empty(M)
        
        for m in range(M):
            theta_m = theta[0][m] if hasattr(theta[0], '__len__') else theta[0]
            
            # Gumbel copula h-function
            if theta_m == 1:
                h[m] = u[m]
            else:
                log_u = np.log(u[m])
                log_v = np.log(v[m])
                
                t1 = (-log_u) ** theta_m
                t2 = (-log_v) ** theta_m
                sum_t = t1 + t2
                
                if sum_t > 0 and log_v != 0:
                    copula_val = np.exp(-sum_t ** (1.0 / theta_m))
                    h[m] = -(copula_val * (sum_t ** (1.0/theta_m - 1.0)) * t2) / (v[m] * log_v)
                else:
                    h[m] = 0

        # Clip output for numerical stability
        h = np.clip(h, UMIN, UMAX)
        return h

    def hinv(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int) -> np.ndarray:
        """
        Inverse conditional distribution function h^(-1)(u|v) for the bivariate Gumbel copula.
        Implementation based on vinecopulib package.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (np.ndarray or float): Gumbel parameter(s), shape (n,) or scalar.
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
        
        for m in range(M):
            theta_m = theta[0][m] if hasattr(theta[0], '__len__') else theta[0]
            
            
            def h_minus_u(x):
                # Calculate h-function and subtract target u
                return self.hfunc(np.array([u[m]]), np.array([v[m]]), np.array([[theta_m]]), 0)[0] - u[m]
            
            try:
                # Solve h(x|v) = u for x
                hinv[m] = brentq(h_minus_u, 1e-12, 1-1e-12)
            except:
                hinv[m] = u[m]  # Fallback

        # Clip output for numerical stability
        hinv = np.clip(hinv, UMIN, UMAX)
        return hinv


    def get_regularization_size(self, dim: int) -> int:
        return dim
    

##########################################################
### Functions for the derivatives and log-likelihood ####
##########################################################


def _log_likelihood(y, theta):
    """
    Log-likelihood for the Gumbel copula.
    """
    M = y.shape[0]
    f = np.empty(M)
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    y_clipped = np.clip(y, UMIN, UMAX)

    for m in range(M):
        if M == 1:
            theta_m = theta
        else: 
            theta_m = theta[m]
            
        u = y_clipped[m, 0]
        v = y_clipped[m, 1]
        
        # Gumbel copula log-likelihood following C implementation
        log_u = np.log(u)
        log_v = np.log(v)

        t1 = (-log_u) ** theta_m + (-log_v) ** theta_m
        
        if t1 > 0 and log_u * log_v > 0:
            f[m] = (-t1 ** (1.0 / theta_m) + 
                (2.0 / theta_m - 2.0) * np.log(t1) + 
                (theta_m - 1.0) * np.log(log_u * log_v) - 
                np.log(u * v) + 
                np.log1p((theta_m - 1.0) * t1 ** (-1.0 / theta_m)))
            
            # Handle numerical limits
            XINFMAX = 700.0  # Approximate maximum for exp
            if f[m] > XINFMAX:
                f[m] = np.log(XINFMAX)
            elif f[m] < np.log(np.finfo(float).tiny):
                f[m] = np.log(np.finfo(float).tiny)
        else:
            f[m] = np.log(np.finfo(float).tiny)

        # Clip output for numerical stability
    f = np.clip(f, UMIN, UMAX)
    return f

def _derivative_1st(y, theta):
    """
    First derivative of the Gumbel copula log-likelihood with respect to theta.
    """
    M = y.shape[0]
    deriv = np.empty(M, dtype=np.float64)
    eps = np.finfo(float).eps
    y = np.clip(y, eps, 1 - eps)
    
    for m in range(M):
        if M == 1:
            theta_m = theta
        else:
            theta_m = theta[m]
        
        u = y[m, 0]
        v = y[m, 1]

        t1 = np.log(u)
        t2 = np.power(-t1, theta_m)
        t3 = np.log(v)
        t4 = np.power(-t3, theta_m)
        t5 = t2 + t4
        t6 = 1.0 / theta_m
        t7 = np.power(t5, t6)
        t8 = theta_m * theta_m
        t10 = np.log(t5)
        t11 = (1.0 / t8) * t10
        t12 = np.log(-t1)
        t14 = np.log(-t3)
        t16 = t2 * t12 + t4 * t14
        t18 = 1.0 / t5
        t20 = -t11 + t6 * t16 * t18
        t22 = np.exp(-t7)
        t23 = -1.0 + t6
        t24 = np.power(t5, 2.0 * t23)
        t25 = t22 * t24
        t27 = t1 * t3
        t28 = theta_m - 1.0
        t29 = np.power(t27, t28)
        t30 = np.power(t5, -t6)
        t31 = t28 * t30
        t32 = 1.0 + t31
        t34 = 1.0 / u
        t35 = 1.0 / v
        t36 = t34 * t35
        t37 = t29 * t32 * t36
        t45 = t25 * t29
        t46 = np.log(t27)
        
        # Handle potential numerical issues
        if (t5 > 0) and (t27 > 0) and (t32 != 0):
            deriv[m] = (
                (-t7 * t20 * t25 * t37 +
                t25 * (-2.0 * t11 + 2.0 * t23 * t16 * t18) * t37 +
                t45 * t46 * t32 * t36 +
                t45 * (t30 - t31 * t20) * t34 * t35) /
                (t22 * t24 * t29 * t32) * u * v
            )
        else:
            deriv[m] = 0.0
    return deriv

def _derivative_2nd(y, theta): 
    """
    Second derivative of the Gumbel copula log-likelihood with respect to theta.
    """
    M = y.shape[0]
    deriv = np.empty(M, dtype=np.float64)
    eps = np.finfo(float).eps
    y = np.clip(y, eps, 1 - eps)
    
    for m in range(M):
        if M == 1:
            theta_m = theta
        else:
            theta_m = theta[m]
            
        u = y[m, 0]
        v = y[m, 1]
        
        t3 = np.log(u)
        t4 = np.power(-t3, 1.0*theta_m)
        t5 = np.log(v)
        t6 = np.power(-t5, 1.0*theta_m)
        t7 = t4+t6
        t8 = 1/theta_m
        t9 = np.power(t7, 1.0*t8)
        t10 = theta_m*theta_m
        t11 = 1/t10
        t12 = np.log(t7)
        t13 = t11*t12
        t14 = np.log(-t3)
        t16 = np.log(-t5)
        t18 = t4*t14+t6*t16
        t20 = 1/t7
        t22 = -t13+t8*t18*t20
        t23 = t22*t22
        t25 = np.exp(-t9)
        t27 = t25/u
        t29 = 1/v
        t30 = -1.0+t8
        t31 = np.power(t7, 2.0*t30)
        t32 = t29*t31
        t33 = t3*t5
        t34 = theta_m-1.0
        t35 = np.power(t33, 1.0*t34)
        t36 = np.power(t7, -1.0*t8)
        t37 = t34*t36
        t38 = 1.0+t37
        t39 = t35*t38
        t40 = t32*t39
        t44 = 1/t10/theta_m*t12
        t47 = t11*t18*t20
        t49 = t14*t14
        t51 = t16*t16
        t53 = t4*t49+t6*t51
        t56 = t18*t18
        t58 = t7*t7
        t59 = 1/t58
        t61 = 2.0*t44-2.0*t47+t8*t53*t20-t8*t56*t59
        t65 = t9*t9
        t70 = t9*t22*t27
        t74 = -2.0*t13+2.0*t30*t18*t20
        t75 = t74*t35
        t80 = np.log(t33)
        t87 = t36-t37*t22
        t88 = t35*t87
        t17 = t27*t29
        t15 = t74*t74
        t2 = t31*t35
        t1 = t80*t80
        
        # Handle potential numerical issues
        if (t7 > 0) and (t33 > 0) and (t38 != 0):
            deriv[m] = (-t9*t23*t27*t40-t9*t61*t27*t40+t65*t23*t27*t40-2.0*t70*t32*t75*t38
                -2.0*t70*t32*t35*t80*t38-2.0*t70*t32*t88+t17*t31*t15*t39+t17*t31*(4.0*t44-4.0*
                t47+2.0*t30*t53*t20-2.0*t30*t56*t59)*t39+2.0*t27*t32*t75*t80*t38+2.0*t17*t31*
                t74*t88+t17*t2*t1*t38+2.0*t17*t2*t80*t87+t17*t2*(-2.0*t36*t22+t37*t23-
                t37*t61))
        else:
            deriv[m] = 0.0
        
    return deriv