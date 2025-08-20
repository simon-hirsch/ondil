import pytest
from sklearn.utils.estimator_checks import check_estimator

import ondil
import ondil.estimators

EXPECTED_FAILED_CHECKS = {
    "check_sample_weight_equivalence_on_dense_data": "To few data points to test this check in the original sklaern implementation.",
}

# https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator


@pytest.mark.parametrize("scale_inputs", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("method", ["ols", "lasso", "elasticnet"])
@pytest.mark.parametrize("ic", ["aic", "bic", "hqc", "max"])
def test_sklearn_compliance_linear_model(scale_inputs, fit_intercept, method, ic):
    estimator = ondil.estimators.OnlineLinearModel(
        fit_intercept=fit_intercept,
        scale_inputs=scale_inputs,
        method=method,
        ic=ic,
    )
    check_estimator(
        estimator, on_fail="raise", expected_failed_checks=EXPECTED_FAILED_CHECKS
    )


@pytest.mark.parametrize("scale_inputs", [False, True])
@pytest.mark.parametrize("method", ["ols", "lasso", "elasticnet"])
@pytest.mark.parametrize("ic", ["aic", "bic", "hqc", "max"])
def test_sklearn_compliance_online_gamlss(scale_inputs, method, ic):
    estimator = ondil.estimators.OnlineDistributionalRegression(
        scale_inputs=scale_inputs,
        method=method,
        ic=ic,
    )
    check_estimator(
        estimator, on_fail="raise", expected_failed_checks=EXPECTED_FAILED_CHECKS
    )


def test_sklearn_compliance_scaler():
    estimator = ondil.OnlineScaler()
    check_estimator(
        estimator,
        on_fail="raise",
    )
