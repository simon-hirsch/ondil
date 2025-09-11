from unittest.mock import patch

import numpy as np
import pytest
from sklearn.utils._tags import get_tags
from sklearn.utils.estimator_checks import check_estimator

from ondil.estimators import (
    MultivariateOnlineDistributionalRegressionPath,
    OnlineDistributionalRegression,
    OnlineLasso,
    OnlineLinearModel,
)
from ondil.scaler import OnlineScaler

EXPECTED_FAILED_CHECKS = {
    "check_sample_weight_equivalence_on_dense_data": "Insufficient data points to validate this check in the original scikit-learn implementation.",
}

# https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator


@pytest.mark.parametrize("scale_inputs", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("method", ["ols", "lasso", "elasticnet"])
@pytest.mark.parametrize("ic", ["aic", "bic", "hqc", "max"])
def test_sklearn_compliance_linear_model(scale_inputs, fit_intercept, method, ic):
    estimator = OnlineLinearModel(
        fit_intercept=fit_intercept,
        scale_inputs=scale_inputs,
        method=method,
        ic=ic,
    )
    check_estimator(
        estimator, on_fail="raise", expected_failed_checks=EXPECTED_FAILED_CHECKS
    )


@pytest.mark.parametrize("scale_inputs", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("ic", ["aic", "bic", "hqc", "max"])
@pytest.mark.parametrize("lambda_n", [50, 100])
@pytest.mark.parametrize("lambda_eps", [1e-4, 1e-2])
def test_sklearn_compliance_lasso_model(
    scale_inputs: bool,
    fit_intercept: bool,
    ic: str,
    lambda_n: int,
    lambda_eps: float,
):
    estimator = OnlineLasso(
        fit_intercept=fit_intercept,
        scale_inputs=scale_inputs,
        ic=ic,
        lambda_n=lambda_n,
        lambda_eps=lambda_eps,
    )
    check_estimator(
        estimator, on_fail="raise", expected_failed_checks=EXPECTED_FAILED_CHECKS
    )


@pytest.mark.parametrize("scale_inputs", [False, True])
@pytest.mark.parametrize("method", ["ols", "lasso", "elasticnet"])
@pytest.mark.parametrize("ic", ["aic", "bic", "hqc", "max"])
def test_sklearn_compliance_online_gamlss(scale_inputs, method, ic):
    estimator = OnlineDistributionalRegression(
        scale_inputs=scale_inputs,
        method=method,
        ic=ic,
    )
    check_estimator(
        estimator, on_fail="raise", expected_failed_checks=EXPECTED_FAILED_CHECKS
    )


def test_sklearn_compliance_scaler():
    estimator = OnlineScaler()
    check_estimator(
        estimator,
        on_fail="raise",
    )


# Monkey patching _enforce_estimator_tags_y to handle multivariate y
# This is necessary because the original implementation does not handle
# multivariate outputs correctly, leading to issues in the estimator checks.
# since sklearn thinks that y is always 1-D and it is sufficient to
# have a (n, 1) shape for y, while ondil assumes that y is at least bivariate
# in the sense of having at least (n, 2) shape.

# https://github.com/scikit-learn/scikit-learn/blob/da08f3d99194565caaa2b6757a3816eef258cd70/sklearn/utils/estimator_checks.py
# this is a helpful git link to understand the monkey patching


def test_sklearn_compliance_multivariate():
    EXPECTED_FAILED_CHECKS_MV = {
        "check_estimators_dtypes": "Weird data generation does not support multivariate properly.",
        "check_fit2d_1feature": "Does not work with my monkey patched version of scikit-learn _enforce_estimator_tags_y.",
    }

    def _enforce_estimator_tags_y(estimator, y):
        # Estimators with a `requires_positive_y` tag only accept strictly positive
        # data
        tags = get_tags(estimator)
        if tags.target_tags.positive_only:
            # Create strictly positive y. The minimal increment above 0 is 1, as
            # y could be of integer dtype.
            y += 1 + abs(y.min())
        if (
            tags.classifier_tags is not None
            and not tags.classifier_tags.multi_class
            and y.size > 0
        ):
            y = np.where(y == y.min(), y, y.min() + 1)
        # Estimators in mono_output_task_error raise ValueError if y is of 1-D
        # Convert into a 2-D y for those estimators.
        if tags.target_tags.multi_output and not tags.target_tags.single_output:
            y = np.vstack((y, np.roll(y, 2))).transpose()
        return y

    with patch(
        "sklearn.utils.estimator_checks._enforce_estimator_tags_y",
        new=_enforce_estimator_tags_y,
    ):
        _ = check_estimator(
            MultivariateOnlineDistributionalRegressionPath(verbose=0),
            on_fail="raise",
            expected_failed_checks=EXPECTED_FAILED_CHECKS_MV,
        )
