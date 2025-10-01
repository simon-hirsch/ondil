import numpy as np
from ondil.coordinate_descent import online_coordinate_descent_path
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lasso_path


def test_coordinate_descent():
    # Sklearn uses a different stopping criterion than we do
    # Also slightly different definition of alpha compared to lambda
    # Hence you get some inconsistencies around _some_ betas in the path

    # Therefore we check the path and assert that 95% of betas match
    # for each explanatory variable

    X, y = load_diabetes(return_X_y=True)
    X /= X.std(axis=0)
    J = X.shape[1]

    x_gram = X.T @ X
    y_gram = X.T @ y
    tolerance = 1e-8

    lambda_eps = 0.0001
    lambda_n = 200
    lambda_max = np.abs(y_gram).max()
    lambda_grid = np.geomspace(lambda_max, lambda_eps * lambda_max, lambda_n)

    is_regularized = np.repeat(True, J)
    beta_path = np.zeros((lambda_n, J))

    ondil_lasso_path = online_coordinate_descent_path(
        x_gram=x_gram,
        y_gram=y_gram,
        beta_path=beta_path,
        lambda_path=lambda_grid,
        alpha=1.0,
        is_regularized=is_regularized,
        early_stop=0,
        beta_lower_bound=np.repeat(-np.inf, J),
        beta_upper_bound=np.repeat(np.inf, J),
        which_start_value="previous_lambda",
        selection="cyclic",
        tolerance=tolerance,
        max_iterations=1000,
    )[0]

    alphas, sklearn_lasso_path, _ = lasso_path(
        X, y, eps=lambda_eps, n_alphas=lambda_n, precompute=x_gram, tol=tolerance
    )
    sklearn_lasso_path = sklearn_lasso_path.T

    assert np.allclose(alphas, lambda_grid / X.shape[0]), "Lambdas don't match"
    assert np.all(
        np.mean(np.isclose(sklearn_lasso_path - ondil_lasso_path, 0), axis=0) > 0.95
    ), "Betas don't match"


def test_coordinate_descent_bounds():
    X, y = load_diabetes(return_X_y=True)
    X /= X.std(axis=0)
    J = X.shape[1]

    x_gram = X.T @ X
    y_gram = X.T @ y
    tolerance = 1e-8

    lambda_eps = 0.0001
    lambda_n = 200
    lambda_max = np.abs(y_gram).max()
    lambda_grid = np.geomspace(lambda_max, lambda_eps * lambda_max, lambda_n)

    is_regularized = np.repeat(True, J)
    beta_path = np.zeros((lambda_n, J))

    ondil_lasso_path_positive = online_coordinate_descent_path(
        x_gram=x_gram,
        y_gram=y_gram,
        beta_path=beta_path,
        lambda_path=lambda_grid,
        alpha=1.0,
        early_stop=0,
        is_regularized=is_regularized,
        beta_lower_bound=np.repeat(0, J),
        beta_upper_bound=np.repeat(np.inf, J),
        which_start_value="previous_lambda",
        selection="cyclic",
        tolerance=tolerance,
        max_iterations=1000,
    )[0]

    ondil_lasso_path_negative = online_coordinate_descent_path(
        x_gram=x_gram,
        y_gram=y_gram,
        beta_path=beta_path,
        lambda_path=lambda_grid,
        alpha=1.0,
        is_regularized=is_regularized,
        early_stop=0,
        beta_lower_bound=np.repeat(-np.inf, J),
        beta_upper_bound=np.repeat(0, J),
        which_start_value="previous_lambda",
        selection="cyclic",
        tolerance=tolerance,
        max_iterations=1000,
    )[0]

    assert np.all(ondil_lasso_path_negative <= 0), "Path should contain only betas <= 0"
    assert np.all(ondil_lasso_path_positive >= 0), "Path should contain only betas >= 0"
