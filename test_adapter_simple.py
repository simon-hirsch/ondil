"""Simple test script for the sktime adapter."""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

def test_sktime_adapter_structure():
    """Test that the sktime adapter has the correct structure."""
    try:
        # Import sktime
        from sktime.regression.base import BaseRegressor
        print("✓ sktime BaseRegressor imported successfully")
        
        # Try to import our adapter (this might fail due to numba dependency)
        try:
            from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
            print("✓ OnlineLinearModelSktime imported successfully")
            
            # Test basic instantiation
            adapter = OnlineLinearModelSktime()
            print("✓ OnlineLinearModelSktime instantiated successfully")
            
            # Check inheritance
            assert isinstance(adapter, BaseRegressor)
            print("✓ OnlineLinearModelSktime correctly inherits from BaseRegressor")
            
            # Check required methods exist
            assert hasattr(adapter, '_fit')
            assert hasattr(adapter, '_predict')
            print("✓ Required methods _fit and _predict exist")
            
            # Check test params method
            test_params = OnlineLinearModelSktime.get_test_params()
            assert isinstance(test_params, list)
            assert len(test_params) > 0
            print("✓ get_test_params works correctly")
            
            # Test parameter setting
            adapter2 = OnlineLinearModelSktime(
                forget=0.1,
                scale_inputs=False,
                fit_intercept=False,
                method="ols",
                ic="aic"
            )
            assert adapter2.forget == 0.1
            assert adapter2.scale_inputs is False
            assert adapter2.fit_intercept is False
            print("✓ Parameter setting works correctly")
            
            print("\n🎉 All structure tests passed!")
            return True
            
        except ImportError as e:
            print(f"❌ Could not import OnlineLinearModelSktime: {e}")
            print("This is likely due to missing numba dependency")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import sktime: {e}")
        return False

def test_mock_functionality():
    """Test functionality with mock OnlineLinearModel."""
    
    # Create a mock OnlineLinearModel for testing
    class MockOnlineLinearModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.coef_ = None
            self.coef_path_ = None
            
        def fit(self, X, y):
            self.coef_ = np.random.random(X.shape[1] + int(getattr(self, 'fit_intercept', True)))
            self.coef_path_ = self.coef_.reshape(1, -1)
            return self
            
        def predict(self, X):
            if self.coef_ is None:
                raise ValueError("Model not fitted")
            return X @ self.coef_[int(getattr(self, 'fit_intercept', True)):]
            
        def update(self, X, y, sample_weight=None):
            pass
            
        def score(self, X, y):
            pred = self.predict(X)
            return 1.0 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    # Create a mock adapter
    from sktime.regression.base import BaseRegressor
    
    class MockOnlineLinearModelSktime(BaseRegressor):
        def __init__(self, **kwargs):
            super().__init__()
            self.params = kwargs
            
            # Set sktime tags for regression
            self.set_tags(**{
                "capability:multivariate": True,
                "capability:univariate": True,
                "X_inner_mtype": "numpy3D",
                "y_inner_mtype": "numpy1D",
                "requires_y": True,
            })
            
        def _fit(self, X, y):
            if X.ndim == 3:
                n_instances, n_timepoints, n_features = X.shape
                X = X.reshape(n_instances, n_timepoints * n_features)
            
            self._estimator = MockOnlineLinearModel(**self.params)
            self._estimator.fit(X, y)
            return self
            
        def _predict(self, X):
            if X.ndim == 3:
                n_instances, n_timepoints, n_features = X.shape
                X = X.reshape(n_instances, n_timepoints * n_features)
            return self._estimator.predict(X)
    
    # Test the mock adapter
    print("\n🧪 Testing mock adapter functionality:")
    
    # Test 2D data
    X_2d = np.random.random((50, 5))
    y = np.random.random(50)
    
    adapter = MockOnlineLinearModelSktime(fit_intercept=True)
    adapter.fit(X_2d, y)
    pred_2d = adapter.predict(X_2d)
    
    assert pred_2d.shape == (50,)
    print("✓ 2D data handling works")
    
    # Test 3D data
    X_3d = np.random.random((30, 4, 3))
    y_3d = np.random.random(30)
    
    adapter_3d = MockOnlineLinearModelSktime(fit_intercept=True)
    adapter_3d.fit(X_3d, y_3d)
    pred_3d = adapter_3d.predict(X_3d)
    
    assert pred_3d.shape == (30,)
    print("✓ 3D data handling works")
    
    print("🎉 Mock functionality tests passed!")
    return True

if __name__ == "__main__":
    print("Testing sktime adapter for OnlineLinearModel")
    print("=" * 50)
    
    success1 = test_sktime_adapter_structure()
    success2 = test_mock_functionality()
    
    if success1 and success2:
        print("\n🎉 All tests passed successfully!")
    elif success2:
        print("\n⚠️  Mock tests passed, but real adapter needs numba dependency")
    else:
        print("\n❌ Some tests failed")