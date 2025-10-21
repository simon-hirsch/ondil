# Author: Your Name  
# MIT Licence
from typing import Dict

import numpy as np
import scipy.stats as st
from scipy.optimize import brentq

from ..base import Distribution, LinkFunction, CopulaMixin
from ..links import LogShiftValue, Log, KendallsTauToParameterGumbel, Identity, GumbelLink
from ..types import ParameterShapes


class BivariateCopulaGumbel(CopulaMixin, Distribution):

    corresponding_gamlss: str = None
    parameter_names = {0: "theta"}
    parameter_support = {0: (1, np.inf)}
    distribution_support = (0, 1) 
    n_params = len(parameter_names)
    parameter_shape = {
        0: ParameterShapes.SCALAR}
    

    def __init__(
        self,
        link: LinkFunction = Log(),
        param_link: LinkFunction = KendallsTauToParameterGumbel(),
        family_code: int = 401,
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
            rotation=0,  # Default rotation, overridden by family_code logic
        )
        self.family_code = family_code  # gamCopula family code (401, 402, 403, 404)
        self.is_multivariate = True
        self._adr_lower_diag = {0: False}
        self._regularization_allowed = {0: False}
        self._regularization = ""
        self._scoring = "fisher"

    @staticmethod
    def fitted_elements(dim: int):
        return {0: int(dim * (dim - 1) // 2)}

    def get_effective_rotation(theta_values: np.ndarray, family_code: int) -> np.ndarray:
        """
        Vectorized version of get_effective_rotation().
        Accepts an array of theta_values and returns corresponding rotations.
        
        Args:
            theta_values (np.ndarray): Copula parameter values (any shape)
            family_code (int): Family code (401–404)
        
        Returns:
            np.ndarray: Effective rotations (same shape as theta_values)
        """
        theta_values = np.asarray(theta_values)
        rot = np.empty_like(theta_values, dtype=int)

        if family_code == 401:
            rot[:] = np.where(theta_values >= 1, 0, 2)
        elif family_code == 402:
            rot[:] = np.where(theta_values >= 1, 0, 3)
        elif family_code == 403:
            rot[:] = np.where(theta_values >= 1, 1, 2)
        elif family_code == 404:
            rot[:] = np.where(theta_values >= 1, 1, 3)
        else:
            raise ValueError(f"Unsupported family code: {family_code}. Supported codes: 401, 402, 403, 404.")

        return rot

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
        chol = theta[0].copy()
        return chol
    
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
        return _derivative_1st(y, theta_param, self.family_code)

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
        return _derivative_2nd(y, theta_param, self.family_code)

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False):
        theta_param = self.theta_to_params(theta)
        
        deriv = _derivative_1st(
                    y, theta_param,  self.family_code,
                )
        return deriv

    def element_dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False):
        theta_param = self.theta_to_params(theta)
              
        deriv = _derivative_2nd(
                    y, theta_param, self.family_code,
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
        return _log_likelihood(y, theta_param, self.family_code)

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")

    def hfunc(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int, family_code: int) -> np.ndarray:
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

        UMIN = 1e-12
        UMAX = 1 - 1e-12

        theta = np.asarray(theta).copy()      # <- prevents in-place mutation of caller's array

        u = np.clip(u, UMIN, UMAX).reshape(-1, 1)
        v = np.clip(v, UMIN, UMAX).reshape(-1, 1)

        # Swap u and v if un == 1
        if un == 1:
            u, v = v, u

        # Get rotations for all samples
        rotation = get_effective_rotation(theta, family_code)
        
        # Apply rotation transformations vectorized
        u_rot, v_rot = u.copy(), v.copy()
        
        # 180° rotation (survival)
        mask_1 = (rotation == 1)
        u_rot[mask_1] = 1 - u[mask_1]
        v_rot[mask_1] = 1 - v[mask_1]
        
        # 90° rotation
        mask_2 = (rotation == 2)
        v_rot[mask_2] = 1 - v[mask_2]
        theta[mask_2] = -theta[mask_2]
        
        # 270° rotation
        mask_3 = (rotation == 3)
        v_rot[mask_3] = 1 - v[mask_3]
        theta[mask_3] = -theta[mask_3]

        log_u = np.log(u_rot)
        log_v = np.log(v_rot)

        t1 = (-log_u) ** theta
        t2 = (-log_v) ** theta
        sum_t = t1 + t2

        copula_val = np.exp(-sum_t ** (1.0 / theta))
        h = -(copula_val * (sum_t ** (1.0/theta - 1.0)) * t2) / (v_rot * log_v)

        return h.squeeze()

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
def get_effective_rotation(theta_values: np.ndarray, family_code: int) -> np.ndarray:
    """
    Vectorized version of get_effective_rotation().
    Accepts an array of theta_values and returns corresponding rotations.
    
    Args:
        theta_values (np.ndarray): Copula parameter values (any shape)
        family_code (int): Family code (401–404)
    
    Returns:
        np.ndarray: Effective rotations (same shape as theta_values)
    """

    theta_values = np.asarray(theta_values)

    rot = np.empty_like(theta_values, dtype=int)

    if family_code == 401:
        rot[:] = np.where(theta_values >= 0, 0, 2)
    elif family_code == 402:
        rot[:] = np.where(theta_values >= 0, 0, 3)
    elif family_code == 403:
        rot[:] = np.where(theta_values >= 0, 1, 2)
    elif family_code == 404:
        rot[:] = np.where(theta_values >= 0, 1, 3)
    else:
        raise ValueError(f"Unsupported family code: {family_code}. Supported codes: 401, 402, 403, 404.")

    return rot

def _log_likelihood(y, theta, family_code=401):
    """
    Log-likelihood for the Gumbel copula.
    """

    theta = np.asarray(theta).copy()      # <- prevents in-place mutation of caller's array
    UMIN = 1e-12
    UMAX = 1 - 1e-12

    y = np.clip(y, UMIN, UMAX)
    u = y[:, 0].reshape(-1, 1)       
    v = y[:, 1].reshape(-1, 1)

    rotation = get_effective_rotation(theta, family_code)

    u_rot, v_rot = u.copy(), v.copy()
    
    # 180° rotation (survival)
    mask_1 = (rotation == 1)
    u_rot[mask_1] = 1 - u[mask_1]
    v_rot[mask_1] = 1 - v[mask_1]
    
    # 90° rotation
    mask_2 = (rotation == 2)
    u_rot[mask_2] = 1 - u[mask_2]
    theta[mask_2] = -theta[mask_2]
    
    # 270° rotation
    mask_3 = (rotation == 3)
    v_rot[mask_3] = 1 - v[mask_3]
    theta[mask_3] = -theta[mask_3]
        
    # Gumbel copula log-likelihood following C implementation
    log_u = np.log(u_rot)
    log_v = np.log(v_rot)

    t1 = (-log_u) ** theta + (-log_v) ** theta

    f = (-t1 ** (1.0 / theta) + 
         (2.0 / theta - 2.0) * np.log(np.maximum(t1, UMIN)) + 
         (theta - 1.0) * np.log(np.maximum(np.abs(log_u * log_v), UMIN)) - 
         np.log(np.maximum(u_rot * v_rot, UMIN)) + 
         np.log1p(np.maximum((theta - 1.0) * t1 ** (-1.0 / theta), -1 + UMIN)))
    
    # Handle numerical limits
    XINFMAX = 700.0  # Approximate maximum for exp
    mask_high = (f > XINFMAX)
    mask_low = (f < np.log(np.finfo(float).tiny))
    
    f[mask_high] = np.log(XINFMAX)
    f[mask_low] = np.log(np.finfo(float).tiny)

    f = np.where(f == 0, 1e-2, f)
    return f.squeeze()


def _derivative_1st(y, theta, family_code=401):
    """
    First derivative of the Gumbel copula log-likelihood with respect to theta.
    """

    theta = np.asarray(theta).copy()      # <- prevents in-place mutation of caller's array
    sign = np.ones_like(theta)

    y = y.copy()

    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12).reshape(-1, 1)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12).reshape(-1, 1)

    rotation = get_effective_rotation(theta, family_code)
    
    u_rot, v_rot = u.copy(), v.copy()
    
    # 180° rotation (survival)
    mask_1 = (rotation == 1)
    u_rot[mask_1] = 1 - u[mask_1]
    v_rot[mask_1] = 1 - v[mask_1]
    sign[mask_1] = 1.0

    mask_2 = (rotation == 2)
    u_rot[mask_2] = 1 - u[mask_2]
    theta[mask_2] = -theta[mask_2]
    sign[mask_2] = -1.0

    mask_3 = (rotation == 3)
    v_rot[mask_3] = 1 - v[mask_3]
    theta[mask_3] = -theta[mask_3]
    sign[mask_3] = -1.0

    t1 = np.log(u_rot)
    t2 = np.power(-t1, theta)
    t3 = np.log(v_rot)
    t4 = np.power(-t3, theta)
    t5 = t2 + t4
    t6 = 1.0 / theta
    t7 = np.power(t5, t6)
    t8 = theta * theta
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
    t28 = theta - 1.0
    t29 = np.power(t27, t28)
    t30 = np.power(t5, -t6)
    t31 = t28 * t30
    t32 = 1.0 + t31
    t34 = 1.0 / u_rot
    t35 = 1.0 / v_rot
    t36 = t34 * t35
    t37 = t29 * t32 * t36
    t45 = t25 * t29
    t46 = np.log(t27)
        
    # Create mask for valid calculations
    mask = (t5 > 0) & (t27 > 0) & (t32 != 0)
    
    # Initialize derivative array
    deriv = np.zeros_like(theta)
    
    # Calculate derivative only for valid entries
    deriv[mask] = (
        (-t7[mask] * t20[mask] * t25[mask] * t37[mask] +
        t25[mask] * (-2.0 * t11[mask] + 2.0 * t23[mask] * t16[mask] * t18[mask]) * t37[mask] +
        t45[mask] * t46[mask] * t32[mask] * t36[mask] +
        t45[mask] * (t30[mask] - t31[mask] * t20[mask]) * t34[mask] * t35[mask]) /
        (t22[mask] * t24[mask] * t29[mask] * t32[mask]) * u_rot[mask] * v_rot[mask]
    )
    deriv *= sign
    return deriv.squeeze()

def _derivative_2nd(y, theta, family_code=401): 
    """
    Second derivative of the Gumbel copula log-likelihood with respect to theta.
    """
    theta = np.asarray(theta).copy()      # <- prevents in-place mutation of caller's array
    y = np.asarray(y).copy()

    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12).reshape(-1, 1)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12).reshape(-1, 1)

    rotation = get_effective_rotation(theta, family_code)
    print(rotation)
    u_rot, v_rot = u.copy(), v.copy()
    
    # 180° rotation (survival)
    mask_1 = (rotation == 1)
    u_rot[mask_1] = 1 - u_rot[mask_1]
    v_rot[mask_1] = 1 - v_rot[mask_1]

    # 90° rotation
    mask_2 = (rotation == 2)
    v_rot[mask_2] = 1 - v_rot[mask_2]
    theta[mask_2] = -theta[mask_2]

    # 270° rotation
    mask_3 = (rotation == 3)
    v_rot[mask_3] = 1 - v_rot[mask_3]
    theta[mask_3] = -theta[mask_3]


    t3 = np.log(np.maximum(u_rot, 1e-12))
    t4 = np.power(-t3, 1.0*theta)
    t5 = np.log(np.maximum(v_rot, 1e-12))
    t6 = np.power(-t5, 1.0*theta)
    t7 = t4+t6
    t8 = 1/theta
    t9 = np.power(np.maximum(t7, 1e-12), 1.0*t8)
    t10 = theta*theta
    t11 = 1/t10
    t12 = np.log(np.maximum(t7, 1e-12))
    t13 = t11*t12
    t14 = np.log(np.maximum(-t3, 1e-12))
    t16 = np.log(np.maximum(-t5, 1e-12))
    t18 = t4*t14+t6*t16
    t20 = 1/np.maximum(t7, 1e-12)
    t22 = -t13+t8*t18*t20
    t23 = t22*t22
    t25 = np.exp(-t9)
    t27 = t25/u_rot
    t29 = 1/v_rot
    t30 = -1.0+t8
    t31 = np.power(np.maximum(t7, 1e-12), 2.0*t30)
    t32 = t29*t31
    t33 = t3*t5
    t34 = theta-1.0
    t35 = np.power(np.maximum(np.abs(t33), 1e-12), 1.0*t34)
    t36 = np.power(np.maximum(t7, 1e-12), -1.0*t8)
    t37 = t34*t36
    t38 = 1.0+t37
    t39 = t35*t38
    t40 = t32*t39
    t44 = 1/t10/theta*t12
    t47 = t11*t18*t20
    t49 = t14*t14
    t51 = t16*t16
    t53 = t4*t49+t6*t51
    t56 = t18*t18
    t58 = t7*t7
    t59 = 1/np.maximum(t58, 1e-12)
    t61 = 2.0*t44-2.0*t47+t8*t53*t20-t8*t56*t59
    t65 = t9*t9
    t70 = t9*t22*t27
    t74 = -2.0*t13+2.0*t30*t18*t20
    t75 = t74*t35
    t80 = np.log(np.maximum(np.abs(t33), 1e-12))
    t87 = t36-t37*t22
    t88 = t35*t87
    t17 = t27*t29
    t15 = t74*t74
    t2 = t31*t35
    t1 = t80*t80
        
    deriv = (-t9*t23*t27*t40-t9*t61*t27*t40+t65*t23*t27*t40-2.0*t70*t32*t75*t38
        -2.0*t70*t32*t35*t80*t38-2.0*t70*t32*t88+t17*t31*t15*t39+t17*t31*(4.0*t44-4.0*
        t47+2.0*t30*t53*t20-2.0*t30*t56*t59)*t39+2.0*t27*t32*t75*t80*t38+2.0*t17*t31*
        t74*t88+t17*t2*t1*t38+2.0*t17*t2*t80*t87+t17*t2*(-2.0*t36*t22+t37*t23-
        t37*t61))
        
    return deriv.squeeze()