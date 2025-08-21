from itertools import product
from typing import Dict

import numpy as np
import scipy.special as sp
import scipy.stats as st

from ..base import Distribution, LinkFunction, MultivariateDistributionMixin
from ..links import Identity, Log, LogShiftTwo, MatrixDiag
from ..types import ParameterShapes


class MultivariateStudentTInverseLowRank(MultivariateDistributionMixin, Distribution):
    """
    The multivariate \\( t \\)-distribution using a low-rank approximation (LRA) of the precision (inverse scale) matrix.

    The PDF of the multivariate \\( t \\)-distribution is given by:
    $$
    p(y \mid \\mu, D, V, \\nu) =
    \\frac{\\Gamma\\left(\\frac{\\nu + k}{2}\\right)}
         {\\Gamma\\left(\\frac{\\nu}{2}\\right) \\left(\\pi \\nu\\right)^{k/2}}
    \\cdot \\frac{1}{\\sqrt{\\det(D + V V^T)}}
    \\left(1 + \\frac{1}{\\nu} (y - \\mu)^T (D + V V^T)^{-1} (y - \\mu)\\right)^{-\\frac{\\nu + k}{2}}
    $$

    where \\( k \\) is the dimensionality of the data, \\( \\mu \\) is the location parameter, \\( D \\) is a diagonal matrix, \\( V \\) is a low-rank matrix, and \\( \\nu \\) is the degrees of freedom.
    """

    corresponding_gamlss: str = None
    parameter_names = {0: "mu", 1: "D", 2: "V", 3: "nu"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (0, np.inf),
        2: (-np.inf, np.inf),
        3: (0, np.inf),
    }
    distribution_support = (-np.inf, np.inf)
    scipy_dist = st.multivariate_t
    scipy_names = {"mu": "loc", "sigma": "shape", "nu": "dof"}
    parameter_shape = {
        0: ParameterShapes.VECTOR,
        1: ParameterShapes.DIAGONAL_MATRIX,
        2: ParameterShapes.MATRIX,
        3: ParameterShapes.SCALAR,
    }

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link_1: LinkFunction = MatrixDiag(Log()),
        scale_link_2: LinkFunction = Identity(),
        tail_link: LinkFunction = LogShiftTwo(),
        rank: int = 3,
        dof_guesstimate: float = 10.0,
    ):
        super().__init__(
            links={0: loc_link, 1: scale_link_1, 2: scale_link_2, 3: tail_link}
        )
        self.rank = rank
        self.dof_guesstimate = dof_guesstimate
        self.dof_independence = 1e6
        self._regularization_allowed = {0: False, 1: False, 2: True, 3: False}

    def fitted_elements(self, dim: int):
        return {0: dim, 1: dim, 2: dim * self.rank, 3: 1}

    @property
    def param_structure(self):
        return self._param_structure

    def index_flat_to_cube(self, k: int, d: int, param: int):
        if (param == 0) | (param == 3):
            return k
        if param == 1:
            i, j = np.diag_indices(d)
            return i[k], j[k]
        if param == 2:
            idx = [(j, i) for i, j in product(range(self.rank), range(d))]
            return idx[k][0], idx[k][1]

    def get_adr_regularization_distance(self, dim: int, param: int) -> np.ndarray:
        if param in (0, 1, 3):
            return None
        if param == 2:
            return np.concatenate([np.repeat(i + 1, dim) for i in range(self.rank)])

    def get_regularization_size(self, dim: int) -> int:
        return self.rank + 1

    def set_theta_element(
        self, theta: Dict, value: np.ndarray, param: int, k: int
    ) -> Dict:
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
        if (param == 0) | (param == 3):
            theta[param][:, k] = value
        if (param == 1) | (param == 2):
            d = theta[0].shape[1]
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            theta[param][:, i, j] = value
        return theta

    def theta_to_params(self, theta):
        loc = theta[0]
        mat_d = theta[1]
        mat_v = theta[2]
        mat_dof = theta[3].squeeze()
        return loc, mat_d, mat_v, mat_dof

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = False
    ):
        mu, mat_d, mat_v, dof = self.theta_to_params(theta)
        d = y.shape[1]
        if param == 0:
            deriv = mv_t_lr_partial_1_mu(y, mu, mat_d, mat_v, dof, i=k)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = mv_t_lr_partial_1_D_element(y, mu, mat_d, mat_v, dof, i=i)
        if param == 2:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = mv_t_lr_partial_1_V_element(y, mu, mat_d, mat_v, dof, i=i, j=j)
        if param == 3:
            deriv = mv_t_lr_partial_1_dof(y, mu, mat_d, mat_v, dof)
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = True
    ):
        mu, mat_d, mat_v, dof = self.theta_to_params(theta)
        d = y.shape[1]
        if param == 0:
            deriv = mv_t_lr_partial_2_mu(y, mu, mat_d, mat_v, dof, i=k)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = mv_t_lr_partial_2_D_element(y, mu, mat_d, mat_v, dof, i=i)
        if param == 2:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = mv_t_lr_partial_2_V_element(y, mu, mat_d, mat_v, dof, i=i, j=j)
        if param == 3:
            deriv = mv_t_lr_partial_2_dof(y, mu, mat_d, mat_v, dof)

        if clip:
            deriv = np.clip(deriv, -np.inf, -1e-10)

        return deriv

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_link(y, i=i, j=j)

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2) | (param == 3):
            return self.links[param].inverse(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_inverse(y, i=i, j=j)

    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2) | (param == 3):
            return self.links[param].link_derivative(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_link_derivative(y, i=i, j=j)

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2) | (param == 3):
            return self.links[param].link_second_derivative(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_link_second_derivative(y, i=i, j=j)

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2) | (param == 3):
            return self.links[param].inverse_derivative(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_inverse_derivative(y, i=i, j=j)

    def initial_values(self, y: np.ndarray, param: int = 0):
        M = y.shape[0]
        if param == 0:
            return np.tile(np.mean(y, axis=0), (M, 1))
        if param == 1:
            mat_d = np.diag(
                1 / (np.var(y, 0) * (self.dof_guesstimate - 2) / self.dof_guesstimate)
            )
            return np.tile(mat_d, (M, 1, 1))
        if param == 2:
            omega = np.linalg.inv(np.cov(y, rowvar=False))
            mat_d = np.diag(
                1 / (np.var(y, 0) * (self.dof_guesstimate - 2) / self.dof_guesstimate)
            )
            eig = np.linalg.eig(omega - mat_d)
            largest_ev = np.argsort(eig.eigenvalues)[-self.rank :][::-1]
            mat_v = eig.eigenvectors[:, largest_ev]
            return np.tile(mat_v, (M, 1, 1))
        if param == 3:
            return np.full((M, 1), self.dof_independence)

    def set_initial_guess(
        self,
        y: np.ndarray,
        theta: Dict[int, np.ndarray],
        param: int,
    ) -> Dict[int, np.ndarray]:
        if param in (0, 2):
            return theta
        if param == 1:
            residuals = y - theta[0]
            variance = np.var(residuals, 0)
            mat_d = np.diag(
                1 / (variance * (self.dof_guesstimate - 2) / self.dof_guesstimate)
            )
            theta[1] = np.tile(mat_d, (y.shape[0], 1, 1))
            return theta
        if param == 3:
            theta[3] = np.full_like(theta[3], self.dof_guesstimate)
            return theta

    def cube_to_flat(self, x: np.ndarray, param: int):
        if (param == 0) | (param == 3):
            return x
        if param == 1:
            return np.copy(x.diagonal(axis1=1, axis2=2))
        if param == 2:
            return x.swapaxes(-1, -2).reshape((x.shape[0], np.prod(x.shape[1:])))

    def flat_to_cube(self, x: np.ndarray, param: int):
        if (param == 0) | (param == 3):
            return x
        if param == 1:
            d = x.shape[1]
            out = np.zeros((x.shape[0], d, d))
            out[:, np.arange(d), np.arange(d)] = x
            return out
        if param == 2:
            d = int(x.shape[1] // self.rank)
            return x.reshape((x.shape[0], self.rank, d)).transpose(0, 2, 1)

    def theta_to_scipy(self, theta: Dict[int, np.ndarray]):
        out = {
            "loc": theta[0],
            "shape": np.linalg.inv(theta[1] + theta[2] @ theta[2].swapaxes(-1, -2)),
            "df": theta[3],
        }
        return out

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        raise NotImplementedError("Not implemented")

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpdf(self, y, theta):
        loc, mat_d, mat_v, dof = self.theta_to_params(theta)
        return batched_log_lilkelihood_t_precision_low_rank_fast(
            y, loc, mat_d=mat_d, mat_v=mat_v, dof=dof
        )

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")


def batched_log_lilkelihood_t_precision_low_rank_fast(y, mu, mat_d, mat_v, dof):
    k = y.shape[1]
    precision = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    A = np.squeeze(sp.gammaln((dof + k) / 2))
    B = np.squeeze(sp.gammaln(dof / 2))
    C = 1 / 2 * k * np.squeeze(np.log(np.pi * dof))
    Z = np.sum((y - mu) * (precision @ (y - mu)[..., None]).squeeze(), 1)
    part1 = A - B - C
    part2 = 1 / 2 * np.log(np.linalg.det(precision))
    part3 = np.squeeze((dof + k) / 2) * np.log((1 + np.squeeze((1 / dof)) * Z))
    return part1 + part2 - part3


def mv_t_lr_partial_1_mu(y, mat_mu, mat_d, mat_v, dof, i):
    k = y.shape[1]
    precision = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    Z = np.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)
    part1 = (k + dof) / (2 * (Z + dof))
    part2 = 2 * np.sum(precision[:, i, :] * (y - mat_mu), axis=1)
    return part1 * part2


def mv_t_lr_partial_2_mu(y, mat_mu, mat_d, mat_v, dof, i):
    k = y.shape[1]

    precision = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    Z = np.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)
    deriv_1 = 2 * np.sum(precision[:, i, :] * (y - mat_mu), axis=1)
    deriv_2 = 2 * precision[:, i, i]

    part1 = k + dof
    part2 = (Z + dof) * deriv_2 - deriv_1**2
    part3 = 2 * (Z + dof) ** 2

    return -(part1 * part2) / part3


def mv_t_lr_partial_1_D_element(y, mat_mu, mat_d, mat_v, dof, i):
    k = y.shape[1]
    precision = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    Z = np.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)

    # Derivative of the log(det(D + VV^T))
    part_1 = 0.5 * np.linalg.inv(precision)[:, i, i]

    # Derivative of the last term
    part2 = np.squeeze(k + dof) / (2 * (Z + dof.squeeze()))
    part3 = (y[:, i] - mat_mu[:, i]) ** 2
    return part_1 - part2 * part3


def mv_t_lr_partial_2_D_element(y, mat_mu, mat_d, mat_v, dof, i):
    k = y.shape[1]
    precision = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    cov = np.linalg.inv(precision)
    Z = np.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)

    # Derivative for the log(det())
    part1 = -0.5 * cov[:, i, i] ** 2

    # Derivative for the last term
    deriv_lambda_1 = (y[:, i] - mat_mu[:, i]) ** 2
    deriv_lambda_2 = 0

    part2 = k + dof.squeeze()
    part3 = (Z + dof.squeeze()) * deriv_lambda_2 - deriv_lambda_1**2
    part4 = 2 * (Z + dof.squeeze()) ** 2
    return part1 - (part2 * part3) / part4


def mv_t_lr_partial_1_V_element(y, mat_mu, mat_d, mat_v, dof, i, j):
    # Would be nice to calculate only the necessary rows of Omega in the future maybe!
    # For part 2
    # zzT @ V
    # zzT[:, i, :] @ mat_v[:, :, j]
    # select the correct row of zzT before
    # sum(z * z[:, i], axis=-1)
    k = y.shape[1]
    precision = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    Z = np.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)

    # Deriviative for the log(det())
    part1 = np.sum(np.linalg.inv(precision)[:, i, :] * mat_v[:, :, j], axis=1)

    # Derivative for the second part
    deriv = -2 * np.sum(
        (y - mat_mu) * np.expand_dims((y[:, i] - mat_mu[:, i]), -1) * mat_v[:, :, j], -1
    )
    # Factor
    part2 = np.squeeze(k + dof)
    part3 = 2 * (Z + dof.squeeze())
    return part1 + (part2 / part3) * deriv


def mv_t_lr_partial_2_V_element(y, mat_mu, mat_d, mat_v, dof, i, j):
    d = y.shape[1]
    precision = mat_d + mat_v @ mat_v.swapaxes(-1, -2)
    cov = np.linalg.inv(precision)
    Z = np.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)

    # second derivative for the log(det())
    sum1 = 0
    sum2 = 0
    for k, q in product(range(d), range(d)):
        sum1 += cov[:, i, i] * mat_v[:, q, j] * cov[:, q, k] * mat_v[:, k, j]
        sum2 += cov[:, i, q] * mat_v[:, q, j] * cov[:, i, k] * mat_v[:, k, j]
    part1 = cov[:, i, i] - sum1 - sum2

    # derivatives for the second part
    deriv1 = -2 * np.sum(
        (y - mat_mu) * np.expand_dims((y[:, i] - mat_mu[:, i]), -1) * mat_v[:, :, j], -1
    )
    deriv2 = 2 * (y - mat_mu)[:, i] ** 2

    part2 = d + dof.squeeze()
    part3 = (Z + dof.squeeze()) * deriv2 - deriv1**2
    part4 = 2 * (Z + dof.squeeze()) ** 2

    return part1 - (part2 * part3) / part4


def mv_t_lr_partial_1_dof(y, mat_mu, mat_d, mat_v, dof):
    k = y.shape[1]

    precision = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    Z = np.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)

    part1 = -(-dof * sp.digamma((k + dof) / 2) + k + dof * sp.digamma(dof / 2)) / (
        2 * dof
    )
    part2 = (Z * (k + dof)) / (2 * dof * (dof + Z)) - 1 / 2 * np.log((dof + Z) / dof)
    return part1 + part2


def mv_t_lr_partial_2_dof(y, mat_mu, mat_d, mat_v, dof):
    k = y.shape[1]
    precision = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    Z = np.sum((y - mat_mu) * (precision @ (y - mat_mu)[..., None]).squeeze(), 1)
    part1 = (
        1
        / 4
        * (
            (2 * k) / (dof**2)
            + sp.polygamma(1, (dof + k) / 2)
            - sp.polygamma(1, dof / 2)
        )
    )
    part2 = (Z * (dof * Z - k * (2 * dof + Z))) / (2 * dof**2 * (dof + Z) ** 2)
    return part1 + part2
