import numpy as np
import pytest

from ondil.coordinate_descent import (
    online_coordinate_descent,
    online_coordinate_descent_quadratic,
    online_coordinate_descent_quadratic_path,
)
from ondil.distributions import Normal
from ondil.estimators import OnlineStructuredAdditiveDistributionRegressor
from ondil.gram import init_gram, init_y_gram
from ondil.methods import QuadraticPenaltyPath
from ondil.terms import InterceptTerm, PSplineTerm
from ondil.terms.splines import (
    make_bspline_basis,
    make_bspline_knots,
    make_centering_constraint,
    make_demmler_reinsch_transform,
    make_difference_penalty,
)

N_SPLINES = 12
DEGREE = 3
DIFF_ORDER = 2


def _make_term_inputs(n):
    distribution = Normal()
    fitted_values = np.tile([0.0, 1.0], (n, 1))
    sample_weight = np.ones(n)
    estimation_weight = np.ones(n)
    return distribution, fitted_values, sample_weight, estimation_weight


# ----------------------------------------------------------------------------
# Phase 1: Basis & penalty
# ----------------------------------------------------------------------------


def test_basis_shape_and_partition_of_unity():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 500)
    knots = make_bspline_knots(0.0, 10.0, N_SPLINES, DEGREE)
    basis = make_bspline_basis(x, knots, DEGREE)
    assert basis.shape == (500, N_SPLINES)
    # Partition of unity inside the domain
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(basis >= 0)


def test_basis_linear_extrapolation():
    knots = make_bspline_knots(0.0, 1.0, N_SPLINES, DEGREE)
    rng = np.random.default_rng(2)
    coef = rng.normal(size=N_SPLINES)

    # Three equally spaced points beyond each boundary: exact collinearity
    for x_out in (np.array([-0.3, -0.2, -0.1]), np.array([1.1, 1.2, 1.3])):
        basis = make_bspline_basis(x_out, knots, DEGREE)
        f = basis @ coef
        np.testing.assert_allclose(f[1], (f[0] + f[2]) / 2, atol=1e-12)

    # Continuity of value and slope at the boundary
    eps = 1e-6
    for boundary, sign in ((0.0, -1.0), (1.0, 1.0)):
        x_eval = np.array([boundary, boundary + sign * eps])
        basis = make_bspline_basis(x_eval, knots, DEGREE)
        f = basis @ coef
        np.testing.assert_allclose(f[1], f[0], atol=1e-4)


def test_penalty_annihilates_polynomials_and_banded():
    penalty = make_difference_penalty(N_SPLINES, DIFF_ORDER)
    # S is symmetric PSD with Frobenius norm 1
    np.testing.assert_allclose(penalty, penalty.T)
    np.testing.assert_allclose(np.linalg.norm(penalty), 1.0)
    # Annihilates polynomial coefficient sequences up to degree q - 1
    idx = np.arange(N_SPLINES, dtype=float)
    for power in range(DIFF_ORDER):
        np.testing.assert_allclose(penalty @ idx**power, 0.0, atol=1e-12)
    # Bandedness: zero outside |i - j| > q
    i, j = np.meshgrid(idx, idx, indexing="ij")
    assert np.all(penalty[np.abs(i - j) > DIFF_ORDER] == 0)


# ----------------------------------------------------------------------------
# Phase 2: Solver
# ----------------------------------------------------------------------------


def _make_gram_problem(seed=42, n=200, k=N_SPLINES):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    knots = make_bspline_knots(0.0, 1.0, k, DEGREE)
    X = make_bspline_basis(x, knots, DEGREE)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, n)
    w = np.ones(n)
    g = init_gram(X, w, 0.0)
    h = init_y_gram(X, y, w, 0.0).squeeze(-1)
    return g, h


def test_quadratic_cd_matches_direct_solve():
    g, h = _make_gram_problem()
    penalty = make_difference_penalty(N_SPLINES, DIFF_ORDER)
    is_regularized = np.zeros(N_SPLINES, dtype=np.bool_)
    for lam in [1e-4, 1e-1, 1e2, 1e5]:
        beta_cd, _ = online_coordinate_descent_quadratic(
            x_gram=g,
            y_gram=h,
            beta=np.zeros(N_SPLINES),
            penalty_matrix=penalty,
            quadratic_regularization=lam,
            regularization=0.0,
            regularization_weights=None,
            is_regularized=is_regularized,
            alpha=1.0,
            beta_lower_bound=None,
            beta_upper_bound=None,
            tolerance=1e-12,
            max_iterations=100_000,
        )
        beta_direct = np.linalg.solve(g + lam * penalty, h)
        np.testing.assert_allclose(beta_cd, beta_direct, atol=1e-4)


def test_quadratic_cd_with_zero_penalty_matches_plain_cd():
    g, h = _make_gram_problem()
    is_regularized = np.ones(N_SPLINES, dtype=np.bool_)
    zero_penalty = np.zeros((N_SPLINES, N_SPLINES))
    for l1 in [0.0, 0.5, 5.0]:
        beta_quad, _ = online_coordinate_descent_quadratic(
            x_gram=g,
            y_gram=h,
            beta=np.zeros(N_SPLINES),
            penalty_matrix=zero_penalty,
            quadratic_regularization=123.0,
            regularization=l1,
            regularization_weights=None,
            is_regularized=is_regularized,
            alpha=1.0,
            beta_lower_bound=None,
            beta_upper_bound=None,
        )
        beta_plain, _ = online_coordinate_descent(
            x_gram=g,
            y_gram=h,
            beta=np.zeros(N_SPLINES),
            regularization=l1,
            regularization_weights=None,
            is_regularized=is_regularized,
            alpha=1.0,
            beta_lower_bound=None,
            beta_upper_bound=None,
        )
        np.testing.assert_array_equal(beta_quad, beta_plain)


def test_quadratic_cd_path_warm_start():
    # Use the Demmler-Reinsch parameterization (as PSplineTerm does), where the
    # Gramian is the identity and the penalty diagonal, so coordinate descent
    # is well conditioned along the whole path.
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(0, 1, n)
    knots = make_bspline_knots(0.0, 1.0, N_SPLINES, DEGREE)
    basis = make_bspline_basis(x, knots, DEGREE)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, n)
    constraint = make_centering_constraint(basis.mean(axis=0))
    gram_constrained = init_gram(basis @ constraint, np.ones(n), 0.0)
    penalty_constrained = (
        constraint.T @ make_difference_penalty(N_SPLINES, DIFF_ORDER) @ constraint
    )
    transform, eigenvalues = make_demmler_reinsch_transform(
        gram_constrained, penalty_constrained
    )
    X = basis @ constraint @ transform
    g = init_gram(X, np.ones(n), 0.0)
    h = init_y_gram(X, y, np.ones(n), 0.0).squeeze(-1)
    penalty = np.diag(eigenvalues)
    k = X.shape[1]

    lambda_path = np.geomspace(1e10, 1e-10, 25)
    beta_path, _ = online_coordinate_descent_quadratic_path(
        x_gram=g,
        y_gram=h,
        beta_path=np.zeros((25, k)),
        lambda_path=lambda_path,
        penalty_matrix=penalty,
        is_regularized=np.zeros(k, dtype=np.bool_),
        alpha=1.0,
        regularization=0.0,
        regularization_weights=None,
        beta_lower_bound=None,
        beta_upper_bound=None,
        tolerance=1e-12,
        max_iterations=10_000,
    )
    for i, lam in enumerate(lambda_path):
        np.testing.assert_allclose(
            beta_path[i], np.linalg.solve(g + lam * penalty, h), atol=1e-8
        )


def test_edf_monotone_and_bounded():
    g, _ = _make_gram_problem()
    penalty = make_difference_penalty(N_SPLINES, DIFF_ORDER)
    lambda_path = np.geomspace(1e8, 1e-8, 40)
    edf = QuadraticPenaltyPath.effective_degrees_of_freedom(g, penalty, lambda_path)
    # decreasing lambda -> increasing edf
    assert np.all(np.diff(edf) >= -1e-8)
    assert np.all(edf >= DIFF_ORDER - 1e-6)
    assert np.all(edf <= N_SPLINES + 1e-6)


def test_large_lambda_recovers_polynomial():
    g, h = _make_gram_problem()
    penalty = make_difference_penalty(N_SPLINES, DIFF_ORDER)
    lam = 1e10 * np.trace(g) / np.trace(penalty)
    beta = np.linalg.solve(g + lam * penalty, h)
    beta_cd, _ = online_coordinate_descent_quadratic(
        x_gram=g,
        y_gram=h,
        beta=np.zeros(N_SPLINES),
        penalty_matrix=penalty,
        quadratic_regularization=lam,
        regularization=0.0,
        regularization_weights=None,
        is_regularized=np.zeros(N_SPLINES, dtype=np.bool_),
        alpha=1.0,
        beta_lower_bound=None,
        beta_upper_bound=None,
        tolerance=1e-14,
        max_iterations=100_000,
    )
    # Coefficients are (numerically) in the null space of the q-th differences,
    # i.e. a polynomial sequence of degree q - 1 (linear for q = 2).
    second_diff = np.diff(beta_cd, n=DIFF_ORDER)
    assert np.max(np.abs(second_diff)) < 1e-6 * (1 + np.max(np.abs(beta)))


# ----------------------------------------------------------------------------
# Phase 3: PSplineTerm
# ----------------------------------------------------------------------------


def _simulate(n, seed=10):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    f = np.sin(2 * np.pi * x)
    y = f + rng.normal(0, 0.2, n)
    return x[:, None], y, f


def test_pspline_term_fit_and_predict():
    X, y, f = _simulate(500)
    distribution, fitted_values, sample_weight, estimation_weight = _make_term_inputs(
        500
    )
    term = PSplineTerm(feature=0, n_splines=N_SPLINES)._prepare_term()
    term = term.fit(
        X=X,
        y=y - y.mean(),
        fitted_values=fitted_values,
        target_values=y,
        distribution=distribution,
        sample_weight=sample_weight,
        estimation_weight=estimation_weight,
    )
    pred = term.predict_out_of_sample(X=X, distribution=distribution)
    assert pred.shape == (500,)
    centered_f = f - f.mean()
    mse = np.mean((pred - centered_f) ** 2)
    assert mse < 0.05 * np.var(centered_f)
    # Selection diagnostics (constrained parameterization has K - 1 coefficients)
    assert term._state.coef_path_.shape == (term.lambda_n, N_SPLINES - 1)
    assert DIFF_ORDER - 1 - 1e-6 <= term.edf_ <= N_SPLINES - 1 + 1e-6
    assert term.lambda_selected_ > 0


def test_pspline_term_fixed_lambda():
    X, y, _ = _simulate(300)
    distribution, fitted_values, sample_weight, estimation_weight = _make_term_inputs(
        300
    )
    term = PSplineTerm(feature=0, n_splines=N_SPLINES, lambda_=10.0)._prepare_term()
    term = term.fit(
        X=X,
        y=y - y.mean(),
        fitted_values=fitted_values,
        target_values=y,
        distribution=distribution,
        sample_weight=sample_weight,
        estimation_weight=estimation_weight,
    )
    assert term._state.coef_path_.shape == (1, N_SPLINES - 1)
    assert term.lambda_selected_ == 10.0


def test_pspline_term_update_matches_batch_grams():
    n, n0 = 600, 400
    X, y, _ = _simulate(n)
    y = y - y.mean()
    distribution, fitted_values, sample_weight, estimation_weight = _make_term_inputs(n)

    term = PSplineTerm(feature=0, n_splines=N_SPLINES, forget=0.0)._prepare_term()
    term = term.fit(
        X=X[:n0],
        y=y[:n0],
        fitted_values=fitted_values[:n0],
        target_values=y[:n0],
        distribution=distribution,
        sample_weight=sample_weight[:n0],
        estimation_weight=estimation_weight[:n0],
    )
    updated = term.update(
        X=X[n0:],
        y=y[n0:],
        fitted_values=fitted_values[n0:],
        target_values=y[n0:],
        distribution=distribution,
        sample_weight=sample_weight[n0:],
        estimation_weight=estimation_weight[n0:],
    )

    # Immutability: the original term is unchanged
    assert updated is not term
    assert updated._state.n_observations == n
    assert term._state.n_observations == n0

    # Gram updates are exact: same as batch grams on the full design matrix
    # built with the *initial* knots and column means.
    X_mat_full = term.make_design_matrix(X)
    g_direct = init_gram(X_mat_full, np.ones(n), 0.0)
    h_direct = init_y_gram(X_mat_full, y, np.ones(n), 0.0)
    np.testing.assert_allclose(updated._state.g, g_direct, atol=1e-8)
    np.testing.assert_allclose(updated._state.h, h_direct, atol=1e-8)

    # Selected coefficients solve the penalized normal equations
    penalty = updated.make_penalty_matrix()
    beta_direct = np.linalg.solve(
        g_direct + updated.lambda_selected_ * penalty, h_direct.squeeze(-1)
    )
    np.testing.assert_allclose(updated.coef_, beta_direct, atol=1e-4)


def test_pspline_term_extrapolates_linearly_out_of_sample():
    X, y, _ = _simulate(400)
    distribution, fitted_values, sample_weight, estimation_weight = _make_term_inputs(
        400
    )
    term = PSplineTerm(feature=0, n_splines=N_SPLINES)._prepare_term()
    term = term.fit(
        X=X,
        y=y - y.mean(),
        fitted_values=fitted_values,
        target_values=y,
        distribution=distribution,
        sample_weight=sample_weight,
        estimation_weight=estimation_weight,
    )
    X_out = np.array([2.0, 2.5, 3.0])[:, None]
    pred = term.predict_out_of_sample(X=X_out, distribution=distribution)
    np.testing.assert_allclose(pred[1], (pred[0] + pred[2]) / 2, atol=1e-10)


def test_pspline_term_invalid_inputs():
    with pytest.raises(ValueError):
        PSplineTerm(feature=0, n_splines=3, degree=3)._prepare_term().fit(
            X=np.linspace(0, 1, 10)[:, None],
            y=np.zeros(10),
            fitted_values=None,
            target_values=None,
            distribution=None,
            sample_weight=np.ones(10),
            estimation_weight=np.ones(10),
        )
    with pytest.raises(ValueError):
        make_difference_penalty(n_splines=5, diff_order=5)
    with pytest.raises(ValueError):
        make_bspline_knots(0.0, 0.0, 10, 3)


# ----------------------------------------------------------------------------
# Phase 4: Integration with the estimator
# ----------------------------------------------------------------------------


def test_integration_structured_additive_regressor():
    rng = np.random.default_rng(123)
    n = 800
    x = rng.uniform(-2, 2, n)
    mu = np.sin(2 * x)
    sigma = np.exp(-0.5 + 0.3 * x)
    y = mu + sigma * rng.normal(size=n)
    X = x[:, None]

    estimator = OnlineStructuredAdditiveDistributionRegressor(
        distribution=Normal(),
        terms={
            0: [InterceptTerm(), PSplineTerm(feature=0, n_splines=15)],
            1: [InterceptTerm(), PSplineTerm(feature=0, n_splines=15)],
        },
        scale_inputs=True,
    )
    estimator.fit(X, y)
    fitted = estimator.predict(X)
    assert fitted.shape == (n, 2)
    assert np.all(np.isfinite(fitted))
    mse_mu = np.mean((fitted[:, 0] - mu) ** 2)
    assert mse_mu < 0.1 * np.var(mu)

    # Online update + prediction run cleanly
    x_new = rng.uniform(-2, 2, 50)
    y_new = np.sin(2 * x_new) + np.exp(-0.5 + 0.3 * x_new) * rng.normal(size=50)
    estimator.update(x_new[:, None], y_new)
    fitted_new = estimator.predict(X)
    assert np.all(np.isfinite(fitted_new))
    mse_mu_new = np.mean((fitted_new[:, 0] - mu) ** 2)
    assert mse_mu_new < 0.1 * np.var(mu)
