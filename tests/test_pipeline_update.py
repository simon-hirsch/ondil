"""Test the Pipeline.update() method for online learning."""

import numpy as np
import pytest

from ondil.estimators import OnlineLinearModel
from ondil.pipeline import DistributionalRegressionPipeline
from ondil.scaler import OnlineScaler


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1

    # Split into initial fit and update batches
    X_fit = X[:50]
    y_fit = y[:50]
    X_update = X[50:75]
    y_update = y[50:75]
    X_test = X[75:]

    return X_fit, y_fit, X_update, y_update, X_test, y


def test_pipeline_update_basic(synthetic_data):
    """Test that pipeline.update() works with OnlineScaler and OnlineLinearModel."""
    X_fit, y_fit, X_update, y_update, X_test, _ = synthetic_data

    # Create pipeline with scaler and estimator
    pipeline = DistributionalRegressionPipeline([
        ("scaler", OnlineScaler()),
        ("estimator", OnlineLinearModel(method="ols")),
    ])

    # Fit on initial data
    pipeline.fit(X_fit, y_fit)
    predictions_after_fit = pipeline.predict(X_test)

    # Update with new data
    pipeline.update(X_update, y_update)
    predictions_after_update = pipeline.predict(X_test)

    # Predictions should change after update
    assert not np.allclose(predictions_after_fit, predictions_after_update), (
        "Predictions should change after update"
    )


def test_pipeline_update_returns_self(synthetic_data):
    """Test that pipeline.update() returns self for method chaining."""
    X_fit, y_fit, X_update, y_update, X_test, _ = synthetic_data

    pipeline = DistributionalRegressionPipeline([
        ("scaler", OnlineScaler()),
        ("estimator", OnlineLinearModel(method="ols")),
    ])

    pipeline.fit(X_fit, y_fit)
    result = pipeline.update(X_update, y_update)

    # Should return self
    assert result is pipeline, "update() should return self for method chaining"


def test_pipeline_update_method_chaining(synthetic_data):
    """Test that pipeline.update() supports method chaining."""
    X_fit, y_fit, X_update, y_update, X_test, _ = synthetic_data

    pipeline = DistributionalRegressionPipeline([
        ("scaler", OnlineScaler()),
        ("estimator", OnlineLinearModel(method="ols")),
    ])

    # Test method chaining
    predictions = pipeline.fit(X_fit, y_fit).update(X_update, y_update).predict(X_test)

    assert predictions.shape == (X_test.shape[0],), (
        f"Expected shape {(X_test.shape[0],)}, got {predictions.shape}"
    )


def test_pipeline_update_consistent_with_direct_updates(synthetic_data):
    """Test that pipeline updates produce same results as direct estimator updates."""
    X_fit, y_fit, X_update, y_update, X_test, _ = synthetic_data

    # Pipeline approach
    pipeline = DistributionalRegressionPipeline([
        ("scaler", OnlineScaler()),
        ("estimator", OnlineLinearModel(method="ols")),
    ])
    pipeline.fit(X_fit, y_fit)
    pipeline.update(X_update, y_update)

    # Direct approach
    scaler_direct = OnlineScaler()
    estimator_direct = OnlineLinearModel(method="ols")

    scaler_direct.fit(X_fit)
    X_scaled = scaler_direct.transform(X_fit)
    estimator_direct.fit(X_scaled, y_fit)

    # Update directly
    scaler_direct.update(X_update)
    X_update_scaled_after = scaler_direct.transform(X_update)
    estimator_direct.update(X_update_scaled_after, y_update)

    # Predictions should be close (may not be exactly equal due to scaler update)
    pipeline_pred = pipeline.predict(X_test)

    X_test_scaled = scaler_direct.transform(X_test)
    direct_pred = estimator_direct.predict(X_test_scaled)

    assert np.allclose(pipeline_pred, direct_pred, rtol=1e-10), (
        "Pipeline and direct updates should produce same predictions"
    )


def test_pipeline_update_without_scaler(synthetic_data):
    """Test pipeline.update() with only an estimator (no transformer)."""
    X_fit, y_fit, X_update, y_update, X_test, _ = synthetic_data

    # Pipeline with just the estimator
    pipeline = DistributionalRegressionPipeline([
        ("estimator", OnlineLinearModel(method="ols"))
    ])

    pipeline.fit(X_fit, y_fit)
    predictions_before = pipeline.predict(X_test)

    # Update should work even without transformers
    pipeline.update(X_update, y_update)
    predictions_after = pipeline.predict(X_test)

    assert not np.allclose(predictions_before, predictions_after), (
        "Predictions should change after update"
    )


def test_pipeline_update_with_multiple_transformers(synthetic_data):
    """Test pipeline.update() with multiple transformers."""
    X_fit, y_fit, X_update, y_update, X_test, _ = synthetic_data

    pipeline = DistributionalRegressionPipeline([
        ("scaler1", OnlineScaler()),
        ("scaler2", OnlineScaler()),
        ("estimator", OnlineLinearModel(method="ols")),
    ])

    pipeline.fit(X_fit, y_fit)
    predictions_before = pipeline.predict(X_test)

    # Update should propagate through all transformers
    pipeline.update(X_update, y_update)
    predictions_after = pipeline.predict(X_test)

    assert not np.allclose(predictions_before, predictions_after), (
        "Predictions should change after update"
    )


def test_pipeline_update_requires_fitted_pipeline(synthetic_data):
    """Test that pipeline.update() requires the pipeline to be fitted first."""
    X_fit, y_fit, X_update, y_update, _, _ = synthetic_data

    pipeline = DistributionalRegressionPipeline([
        ("scaler", OnlineScaler()),
        ("estimator", OnlineLinearModel(method="ols")),
    ])

    # Should raise error if trying to update unfitted pipeline
    with pytest.raises(Exception):  # sklearn raises NotFittedError
        pipeline.update(X_update, y_update)


def test_pipeline_update_transform_propagation(synthetic_data):
    """Test that data is correctly transformed through pipeline during update."""
    X_fit, y_fit, X_update, y_update, _, _ = synthetic_data

    # Create pipeline with a scaler that has a measurable effect
    pipeline = DistributionalRegressionPipeline([
        ("scaler", OnlineScaler()),
        ("estimator", OnlineLinearModel(method="ols")),
    ])

    pipeline.fit(X_fit, y_fit)

    # Before update, get the scaler state
    mean_before = pipeline.named_steps["scaler"].mean_.copy()

    # After update, scaler should be updated
    pipeline.update(X_update, y_update)
    mean_after = pipeline.named_steps["scaler"].mean_.copy()

    # Scaler should have been updated (mean should change)
    assert not np.allclose(mean_before, mean_after), (
        "Scaler state should change after update"
    )
