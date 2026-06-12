import copy
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

from ..base import Distribution, Term
from ..gram import init_forget_vector
from ..information_criteria import InformationCriterion
from ..methods import get_estimation_method
from ..methods.quadratic_penalty import QuadraticPenaltyPath
from ..utils import calculate_effective_training_length


def make_bspline_knots(
    x_min: float,
    x_max: float,
    n_splines: int,
    degree: int,
    padding: float = 0.0,
) -> np.ndarray:
    r"""Construct an equidistant B-spline knot vector.

    The knot vector has `n_splines + degree + 1` equidistant knots such that
    the domain `[x_min - pad, x_max + pad]` is covered by `n_splines` basis
    functions of the given degree, where `pad = padding * (x_max - x_min)`.

    Args:
        x_min (float): Lower end of the data range.
        x_max (float): Upper end of the data range.
        n_splines (int): Number of B-spline basis functions $K$.
        degree (int): Degree of the B-spline basis $p$.
        padding (float): Relative padding of the data range. Defaults to 0.0.

    Returns:
        np.ndarray: Knot vector of length $K + p + 1$.
    """
    if n_splines <= degree:
        raise ValueError("n_splines must be larger than the spline degree.")
    data_range = x_max - x_min
    if data_range <= 0:
        raise ValueError("Feature has zero range; cannot construct spline basis.")
    pad = padding * data_range
    lower = x_min - pad
    upper = x_max + pad
    step = (upper - lower) / (n_splines - degree)
    return lower + step * np.arange(-degree, n_splines + 1)


def make_bspline_basis(
    x: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    r"""Evaluate the B-spline basis with linear extrapolation outside the boundary knots.

    Inside the domain `[knots[degree], knots[-degree - 1]]` the basis is evaluated
    via `scipy.interpolate.BSpline.design_matrix`. Outside the domain, each basis
    function is extrapolated linearly from its value and first derivative at the
    nearest boundary (mgcv `bs="ps"` behavior).

    Args:
        x (np.ndarray): Points at which to evaluate the basis.
        knots (np.ndarray): Full knot vector.
        degree (int): Degree of the B-spline basis.

    Returns:
        np.ndarray: Basis matrix of shape `(len(x), len(knots) - degree - 1)`.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n_splines = knots.shape[0] - degree - 1
    lower = knots[degree]
    upper = knots[-degree - 1]

    basis = np.zeros((x.shape[0], n_splines))
    inside = (x >= lower) & (x <= upper)
    if np.any(inside):
        basis[inside, :] = BSpline.design_matrix(x[inside], knots, degree).toarray()

    for mask, boundary in ((x < lower, lower), (x > upper, upper)):
        if np.any(mask):
            spline = BSpline(knots, np.eye(n_splines), degree)
            value = spline(boundary)
            slope = spline.derivative()(boundary)
            basis[mask, :] = (
                value[None, :] + (x[mask, None] - boundary) * slope[None, :]
            )

    return basis


def make_difference_penalty(
    n_splines: int,
    diff_order: int,
    normalize: bool = True,
) -> np.ndarray:
    r"""Construct the P-spline difference penalty matrix $S = D_q^\top D_q$.

    Args:
        n_splines (int): Number of B-spline coefficients $K$.
        diff_order (int): Order $q$ of the difference operator.
        normalize (bool): Normalize $S$ by its Frobenius norm so that the
            penalty strength has comparable meaning across terms. Defaults to True.

    Returns:
        np.ndarray: Penalty matrix of shape `(K, K)`.
    """
    if not 0 < diff_order < n_splines:
        raise ValueError("diff_order must be in (0, n_splines).")
    diff = np.diff(np.eye(n_splines), n=diff_order, axis=0)
    penalty = diff.T @ diff
    if normalize:
        penalty = penalty / np.linalg.norm(penalty)
    return penalty


def make_centering_constraint(column_means: np.ndarray) -> np.ndarray:
    r"""Construct the sum-to-zero constraint reparameterization matrix.

    For the constraint $m^\top \beta = 0$ (with $m$ the weighted column means of
    the basis, i.e. the fitted values have zero weighted mean), returns an
    orthonormal basis $Z$ of the null space of $m^\top$ such that $\beta = Z\gamma$.
    This removes the constant component of the smooth, avoiding identifiability
    clashes with an intercept, and renders the penalized Gramian nonsingular.

    Args:
        column_means (np.ndarray): Weighted column means $m$ of the basis matrix.

    Returns:
        np.ndarray: Constraint matrix of shape `(K, K - 1)`.
    """
    q, _ = np.linalg.qr(column_means[:, None], mode="complete")
    return q[:, 1:]


def make_demmler_reinsch_transform(
    gram: np.ndarray,
    penalty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Construct the Demmler-Reinsch reparameterization.

    Finds a transformation $T$ such that, for coefficients $\beta = T\gamma$,
    the Gramian becomes the identity, $T^\top G T = I$, and the penalty becomes
    diagonal, $T^\top S T = \mathrm{diag}(\Lambda)$ with $\Lambda \ge 0$. In this
    parameterization the penalized Gramian $I + \lambda\,\mathrm{diag}(\Lambda)$
    is perfectly conditioned for coordinate descent at any penalty strength.

    Args:
        gram (np.ndarray): The (weighted) Gramian $G$ of the design matrix.
        penalty (np.ndarray): The penalty matrix $S$.

    Returns:
        tuple[np.ndarray, np.ndarray]: The transformation matrix $T$ and the
            penalty eigenvalues $\Lambda$.
    """
    eigenvalues_gram, eigenvectors_gram = np.linalg.eigh(gram)
    eigenvalues_gram = np.maximum(eigenvalues_gram, np.max(eigenvalues_gram) * 1e-10)
    inverse_sqrt = eigenvectors_gram / np.sqrt(eigenvalues_gram)
    middle = inverse_sqrt.T @ penalty @ inverse_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(middle)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    transform = inverse_sqrt @ eigenvectors
    return transform, eigenvalues


@dataclass(frozen=True)
class PSplineTermState:
    knots: np.ndarray | None
    column_means: np.ndarray | None
    transform: np.ndarray | None
    penalty_eigenvalues: np.ndarray | None
    g: np.ndarray | None
    h: np.ndarray | None
    coef_path_: np.ndarray | None
    lambda_grid: np.ndarray | None
    edf_: np.ndarray | None
    ic_values_: np.ndarray | None
    best_idx: int | None
    coef_: np.ndarray | None
    rss: np.ndarray | None
    n_observations: int | None


class PSplineTerm(Term):
    r"""Univariate penalized B-spline (P-spline) term.

    Fits a smooth function of a single feature using a B-spline basis with
    equidistant knots and a difference penalty $S = D_q^\top D_q$ on adjacent
    spline coefficients. The penalty strength $\lambda$ is either fixed or
    selected over a geometric grid by minimizing an information criterion with
    effective degrees of freedom
    $\mathrm{edf}(\lambda) = \mathrm{tr}\big((G + \lambda S)^{-1} G\big)$.

    Knots are fixed from the data range at `fit` time (with configurable
    relative padding); predictions beyond the boundary knots use linear
    extrapolation. Identifiability is ensured by a sum-to-zero constraint on
    the fitted values: the weighted column means of the basis (stored at fit
    time) define the constraint $m^\top\beta = 0$, which is absorbed by an
    orthonormal reparameterization with $K - 1$ coefficients. The term thus
    carries no intercept and does not clash with an `InterceptTerm` in the
    backfitting loop. On top of the constraint, a Demmler-Reinsch
    reparameterization (fixed at fit time) renders the Gramian (approximately)
    the identity and the penalty diagonal, so the coordinate descent solver is
    well conditioned at any penalty strength.

    !!! note
        With `diff_order=2` the penalty leaves a linear trend in the feature
        unpenalized, which overlaps with a `LinearTerm` on the same feature.
        The effective degrees of freedom are per-term (conditional on the other
        terms), not a joint mgcv-style EDF.
    """

    allow_online_updates: bool = True

    def __init__(
        self,
        feature: int,
        n_splines: int = 20,
        degree: int = 3,
        diff_order: int = 2,
        lambda_: float | None = None,
        lambda_n: int = 50,
        lambda_eps: float = 1e-12,
        ic: Literal["aic", "bic", "aicc", "hqc", "max"] = "aic",
        forget: float = 0.0,
        knot_padding: float = 0.05,
        method: QuadraticPenaltyPath | None = None,
    ):
        """
        Args:
            feature (int): Column index of the feature to smooth.
            n_splines (int): Number of B-spline basis functions. Defaults to 20.
            degree (int): Degree of the B-spline basis. Defaults to 3 (cubic).
            diff_order (int): Order of the difference penalty. Defaults to 2.
            lambda_ (float | None): Fixed penalty strength. If None, the strength
                is selected over a grid via the information criterion. Defaults to None.
            lambda_n (int): Number of grid points for the penalty strength. Defaults to 50.
            lambda_eps (float): Ratio of smallest to largest grid value. Defaults to 1e-12.
            ic (Literal["aic", "bic", "aicc", "hqc", "max"]): Information criterion
                for the selection of the penalty strength. Defaults to "aic".
            forget (float): Forgetting factor for online updates. Defaults to 0.0.
            knot_padding (float): Relative padding of the data range used for knot
                placement, guarding against early online updates hitting the
                extrapolation region. Defaults to 0.05.
            method (QuadraticPenaltyPath | None): Estimation method instance.
                If None, a `QuadraticPenaltyPath` is created from the lambda
                parameters above. Defaults to None.
        """
        self.feature = feature
        self.n_splines = n_splines
        self.degree = degree
        self.diff_order = diff_order
        self.lambda_ = lambda_
        self.lambda_n = lambda_n
        self.lambda_eps = lambda_eps
        self.ic = ic
        self.forget = forget
        self.knot_padding = knot_padding
        if method is None:
            method = QuadraticPenaltyPath(
                lambda_n=lambda_n,
                lambda_eps=lambda_eps,
                lambda_=lambda_,
            )
        self.method = method

    def _prepare_term(self):
        self._method = get_estimation_method(self.method)
        if not isinstance(self._method, QuadraticPenaltyPath):
            raise ValueError("PSplineTerm requires a QuadraticPenaltyPath method.")
        return self

    @property
    def coef_(self) -> np.ndarray:
        if not hasattr(self, "_state"):
            raise AttributeError("The term has not been fitted yet.")
        return self._state.coef_

    @property
    def edf_(self) -> float:
        """Effective degrees of freedom of the selected fit."""
        if not hasattr(self, "_state"):
            raise AttributeError("The term has not been fitted yet.")
        return float(self._state.edf_[self._state.best_idx])

    @property
    def lambda_selected_(self) -> float:
        """Selected penalty strength."""
        if not hasattr(self, "_state"):
            raise AttributeError("The term has not been fitted yet.")
        return float(self._state.lambda_grid[self._state.best_idx])

    def make_penalty_matrix(self) -> np.ndarray:
        """Construct the (diagonal) penalty matrix in the fitted parameterization."""
        if not hasattr(self, "_state"):
            raise AttributeError("The term has not been fitted yet.")
        return np.diag(self._state.penalty_eigenvalues)

    def _make_basis(self, X: np.ndarray, knots: np.ndarray) -> np.ndarray:
        x = np.asarray(X)[:, self.feature]
        return make_bspline_basis(x=x, knots=knots, degree=self.degree)

    def make_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Create the reparameterized B-spline design matrix using the fitted knots and transform."""
        if not hasattr(self, "_state"):
            raise AttributeError("The term has not been fitted yet.")
        return self._make_basis(X, self._state.knots) @ self._state.transform

    def make_design_matrix_in_sample_during_fit(self, X: np.ndarray, **kwargs):
        return self.make_design_matrix(X)

    def make_design_matrix_in_sample_during_update(self, X: np.ndarray, **kwargs):
        return self.make_design_matrix(X)

    def make_design_matrix_out_of_sample(self, X: np.ndarray, **kwargs):
        return self.make_design_matrix(X)

    def _model_selection(
        self,
        g: np.ndarray,
        h: np.ndarray,
        coef_path_: np.ndarray,
        penalty: np.ndarray,
        lambda_grid: np.ndarray,
        rss: np.ndarray,
        n_observations: int,
    ):
        edf_ = self._method.effective_degrees_of_freedom(
            x_gram=g,
            penalty_matrix=penalty,
            lambda_path=lambda_grid,
        )
        n_effective = calculate_effective_training_length(self.forget, n_observations)
        ic_values_ = InformationCriterion(
            n_observations=n_effective,
            n_parameters=edf_,
            criterion=self.ic,
        ).from_rss(rss)
        best_idx = int(np.argmin(ic_values_))
        return edf_, ic_values_, best_idx

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "PSplineTerm":
        """Fit the P-spline term on the working vector `y`."""
        x = np.asarray(X)[:, self.feature]
        knots = make_bspline_knots(
            x_min=float(np.min(x)),
            x_max=float(np.max(x)),
            n_splines=self.n_splines,
            degree=self.degree,
            padding=self.knot_padding,
        )
        basis = make_bspline_basis(x=x, knots=knots, degree=self.degree)

        forget_weights = init_forget_vector(self.forget, y.shape[0])
        column_means = np.average(basis, weights=sample_weight * forget_weights, axis=0)
        constraint = make_centering_constraint(column_means)

        # Demmler-Reinsch reparameterization of the constrained basis:
        # identity Gramian, diagonal penalty.
        gram_constrained = self._method.init_x_gram(
            X=basis @ constraint,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        penalty_constrained = (
            constraint.T
            @ make_difference_penalty(
                n_splines=self.n_splines,
                diff_order=self.diff_order,
            )
            @ constraint
        )
        transform_dr, penalty_eigenvalues = make_demmler_reinsch_transform(
            gram=gram_constrained,
            penalty=penalty_constrained,
        )
        transform = constraint @ transform_dr
        X_mat = basis @ transform

        penalty = np.diag(penalty_eigenvalues)
        is_regularized = np.zeros(X_mat.shape[1], dtype=np.bool_)

        g = self._method.init_x_gram(
            X=X_mat,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        h = self._method.init_y_gram(
            X=X_mat,
            y=y,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        coef_path_ = self._method.fit_beta_path(
            x_gram=g,
            y_gram=h,
            is_regularized=is_regularized,
            penalty_matrix=penalty,
        )
        lambda_grid = self._method.lambda_path_

        n_observations = y.shape[0]
        residuals = y[:, None] - X_mat @ coef_path_.T
        rss = np.sum((residuals**2) * (sample_weight * forget_weights)[:, None], axis=0)
        rss = rss / np.mean(sample_weight * forget_weights)

        edf_, ic_values_, best_idx = self._model_selection(
            g=g,
            h=h,
            coef_path_=coef_path_,
            penalty=penalty,
            lambda_grid=lambda_grid,
            rss=rss,
            n_observations=n_observations,
        )

        self._state = PSplineTermState(
            knots=knots,
            column_means=column_means,
            transform=transform,
            penalty_eigenvalues=penalty_eigenvalues,
            g=g,
            h=h,
            coef_path_=coef_path_,
            lambda_grid=lambda_grid,
            edf_=edf_,
            ic_values_=ic_values_,
            best_idx=best_idx,
            coef_=coef_path_[best_idx, :],
            rss=rss,
            n_observations=n_observations,
        )
        return self

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitted_values: np.ndarray,
        target_values: np.ndarray,
        distribution: Distribution,
        sample_weight: np.ndarray,
        estimation_weight: np.ndarray,
    ) -> "PSplineTerm":
        """Update the P-spline term with new data, keeping knots and constraint fixed."""
        X_mat = self.make_design_matrix(X)
        penalty = self.make_penalty_matrix()
        is_regularized = np.zeros(X_mat.shape[1], dtype=np.bool_)

        g = self._method.update_x_gram(
            gram=self._state.g,
            X=X_mat,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        h = self._method.update_y_gram(
            gram=self._state.h,
            X=X_mat,
            y=y,
            weights=sample_weight * estimation_weight,
            forget=self.forget,
        )
        coef_path_ = self._method.update_beta_path(
            x_gram=g,
            y_gram=h,
            beta_path=self._state.coef_path_,
            is_regularized=is_regularized,
            penalty_matrix=penalty,
        )
        lambda_grid = self._method.lambda_path_

        n_observations = self._state.n_observations + y.shape[0]
        forget_weights = init_forget_vector(self.forget, y.shape[0])
        residuals = y[:, None] - X_mat @ coef_path_.T
        rss_new = np.sum(
            (residuals**2) * (sample_weight * forget_weights)[:, None], axis=0
        )
        rss_new = rss_new / np.mean(sample_weight * forget_weights)
        rss = (1 - self.forget) ** y.shape[0] * self._state.rss + rss_new

        edf_, ic_values_, best_idx = self._model_selection(
            g=g,
            h=h,
            coef_path_=coef_path_,
            penalty=penalty,
            lambda_grid=lambda_grid,
            rss=rss,
            n_observations=n_observations,
        )

        new_instance = copy.copy(self)
        new_instance._state = replace(
            self._state,
            g=g,
            h=h,
            coef_path_=coef_path_,
            lambda_grid=lambda_grid,
            edf_=edf_,
            ic_values_=ic_values_,
            best_idx=best_idx,
            coef_=coef_path_[best_idx, :],
            rss=rss,
            n_observations=n_observations,
        )
        return new_instance

    def predict_out_of_sample(
        self,
        X: np.ndarray,
        distribution: Distribution,
    ) -> np.ndarray:
        """Predict the smooth contribution for new data."""
        X_mat = self.make_design_matrix(X)
        return X_mat @ self._state.coef_
