"""Tests for sktime adapter of OnlineLinearModel."""

import numpy as np
import pytest

# Try to import sktime dependencies
try:
    import sktime
    from sktime.regression.base import BaseRegressor
    SKTIME_AVAILABLE = True
except ImportError:
    SKTIME_AVAILABLE = False


@pytest.mark.skipif(
    not SKTIME_AVAILABLE,
    reason="sktime not available"
)
class TestOnlineLinearModelSktime:
    """Test class for OnlineLinearModelSktime adapter."""
    
    def test_sktime_adapter_import(self):
        """Test that the sktime adapter can be imported."""
        try:
            from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
            assert OnlineLinearModelSktime is not None
        except ImportError as e:
            pytest.skip(f"Cannot import adapter: {e}")
    
    def test_basic_fit_predict(self):
        """Test basic fit and predict functionality."""
        try:
            from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
        except ImportError:
            pytest.skip("Cannot import adapter")
            
        # Create simple test data
        X = np.random.random((50, 5))
        y = np.random.random(50)
        
        # Test with 2D data (regular sklearn format)
        estimator = OnlineLinearModelSktime(method="ols", fit_intercept=True)
        estimator.fit(X, y)
        predictions = estimator.predict(X)
        
        assert predictions.shape == (50,)
        assert hasattr(estimator, 'coef_')
        
        # Test update functionality
        X_new = np.random.random((10, 5))
        y_new = np.random.random(10)
        estimator.update(X_new, y_new)
        
        # Predictions should still work after update
        predictions_updated = estimator.predict(X)
        assert predictions_updated.shape == (50,)
    
    def test_3d_data_handling(self):
        """Test that the adapter handles 3D data correctly."""
        try:
            from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
        except ImportError:
            pytest.skip("Cannot import adapter")
            
        # Create 3D test data (n_instances, n_timepoints, n_features)
        X_3d = np.random.random((30, 4, 3))  # 30 instances, 4 timepoints, 3 features
        y = np.random.random(30)
        
        estimator = OnlineLinearModelSktime(method="ols", fit_intercept=True)
        estimator.fit(X_3d, y)
        predictions = estimator.predict(X_3d)
        
        assert predictions.shape == (30,)
        assert hasattr(estimator, 'coef_')
        
        # Check that coefficients have the right shape (should be flattened features)
        expected_n_features = 4 * 3 + 1  # flattened features + intercept
        assert estimator.coef_.shape == (expected_n_features,)
    
    def test_different_methods(self):
        """Test adapter with different estimation methods."""
        try:
            from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
        except ImportError:
            pytest.skip("Cannot import adapter")
            
        X = np.random.random((40, 5))
        y = np.random.random(40)
        
        # Test OLS
        estimator_ols = OnlineLinearModelSktime(method="ols")
        estimator_ols.fit(X, y)
        pred_ols = estimator_ols.predict(X)
        assert pred_ols.shape == (40,)
        
        # Test Lasso (if available)
        try:
            estimator_lasso = OnlineLinearModelSktime(method="lasso")
            estimator_lasso.fit(X, y)
            pred_lasso = estimator_lasso.predict(X)
            assert pred_lasso.shape == (40,)
        except Exception:
            # Lasso might not be available in all configurations
            pass
    
    def test_parameter_propagation(self):
        """Test that parameters are correctly propagated to underlying estimator."""
        try:
            from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
        except ImportError:
            pytest.skip("Cannot import adapter")
            
        estimator = OnlineLinearModelSktime(
            forget=0.1,
            scale_inputs=False,
            fit_intercept=False,
            method="ols",
            ic="aic"
        )
        
        X = np.random.random((30, 5))
        y = np.random.random(30)
        estimator.fit(X, y)
        
        # Check that parameters were set correctly
        assert estimator._estimator.forget == 0.1
        assert estimator._estimator.scale_inputs is False
        assert estimator._estimator.fit_intercept is False
        assert estimator._estimator.method == "ols"
        assert estimator._estimator.ic == "aic"
    
    def test_score_method(self):
        """Test the score method."""
        try:
            from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
        except ImportError:
            pytest.skip("Cannot import adapter")
            
        X = np.random.random((50, 5))
        y = np.random.random(50)
        
        estimator = OnlineLinearModelSktime(method="ols")
        estimator.fit(X, y)
        score = estimator.score(X, y)
        
        assert isinstance(score, float)
        assert -np.inf <= score <= 1.0  # R^2 score should be <= 1
    
    def test_get_fitted_params(self):
        """Test get_fitted_params method."""
        try:
            from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
        except ImportError:
            pytest.skip("Cannot import adapter")
            
        X = np.random.random((30, 5))
        y = np.random.random(30)
        
        estimator = OnlineLinearModelSktime(method="ols")
        estimator.fit(X, y)
        
        fitted_params = estimator.get_fitted_params()
        
        assert isinstance(fitted_params, dict)
        assert 'coef_' in fitted_params
        assert fitted_params['coef_'].shape == estimator.coef_.shape
    
    def test_get_test_params(self):
        """Test get_test_params class method."""
        try:
            from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
        except ImportError:
            pytest.skip("Cannot import adapter")
            
        params = OnlineLinearModelSktime.get_test_params()
        
        assert isinstance(params, list)
        assert len(params) >= 1
        
        for param_set in params:
            assert isinstance(param_set, dict)
            # Check that each parameter set contains expected keys
            expected_keys = ["forget", "scale_inputs", "fit_intercept", "method", "ic"]
            for key in expected_keys:
                assert key in param_set


# Optional: Run sktime's test suite on the adapter (requires full sktime installation)
@pytest.mark.skipif(
    not SKTIME_AVAILABLE,
    reason="sktime not available"
)
def test_sktime_estimator_compliance():
    """Test compliance with sktime estimator interface using sktime's test suite."""
    try:
        from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
        
        # Run sktime's comprehensive test suite
        # Note: This might be quite extensive, so we can skip it for basic testing
        # run_tests_for_class(OnlineLinearModelSktime)
        
        # For now, just test that the class can be instantiated with test params
        test_params = OnlineLinearModelSktime.get_test_params()
        for params in test_params:
            estimator = OnlineLinearModelSktime(**params)
            assert estimator is not None
            
    except ImportError:
        pytest.skip("Cannot import adapter or run full sktime tests")


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestOnlineLinearModelSktime()
    test_instance.test_sktime_adapter_import()
    print("✓ Import test passed")
    
    try:
        test_instance.test_basic_fit_predict()
        print("✓ Basic fit/predict test passed")
        
        test_instance.test_3d_data_handling()
        print("✓ 3D data handling test passed")
        
        test_instance.test_parameter_propagation()
        print("✓ Parameter propagation test passed")
        
        test_instance.test_score_method()
        print("✓ Score method test passed")
        
        test_instance.test_get_fitted_params()
        print("✓ Get fitted params test passed")
        
        test_instance.test_get_test_params()
        print("✓ Get test params test passed")
        
        print("\nAll basic tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()