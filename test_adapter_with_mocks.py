#!/usr/bin/env python3
"""Test script that creates minimal mocks to test the sktime adapter."""

import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_mock_ondil_dependencies():
    """Create mock modules for ondil dependencies."""
    
    # Mock numba
    mock_numba = MagicMock()
    sys.modules['numba'] = mock_numba
    
    # Mock the gram module 
    mock_gram = MagicMock()
    mock_gram.init_forget_vector = lambda forget, n: np.ones(n)
    sys.modules['ondil.gram'] = mock_gram
    
    # Mock other dependencies
    sys.modules['ondil.base'] = MagicMock()
    sys.modules['ondil.design_matrix'] = MagicMock()
    sys.modules['ondil.information_criteria'] = MagicMock()
    sys.modules['ondil.methods'] = MagicMock()
    sys.modules['ondil.scaler'] = MagicMock()
    sys.modules['ondil.utils'] = MagicMock()
    
    # Mock the OnlineLinearModel with a realistic implementation
    class MockOnlineLinearModel:
        def __init__(self, forget=0.0, scale_inputs=True, fit_intercept=True, 
                     regularize_intercept=False, method="ols", ic="bic"):
            self.forget = forget
            self.scale_inputs = scale_inputs
            self.fit_intercept = fit_intercept
            self.regularize_intercept = regularize_intercept
            self.method = method
            self.ic = ic
            self.coef_ = None
            self.coef_path_ = None
            self.n_observations_ = None
            self.n_features_ = None
            
        def fit(self, X, y, sample_weight=None):
            n_samples, n_features = X.shape
            self.n_observations_ = n_samples
            self.n_features_ = n_features + int(self.fit_intercept)
            
            # Create realistic coefficients
            self.coef_ = np.random.random(self.n_features_) * 0.1
            self.coef_path_ = self.coef_.reshape(1, -1)
            return self
            
        def predict(self, X):
            if self.coef_ is None:
                raise ValueError("Model not fitted")
            
            # Add intercept if needed
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(X.shape[0]), X])
            else:
                X_design = X
                
            return X_design @ self.coef_
            
        def update(self, X, y, sample_weight=None):
            # Mock update - just add some noise to coefficients
            if self.coef_ is not None:
                self.coef_ += np.random.normal(0, 0.01, self.coef_.shape)
            return self
            
        def score(self, X, y):
            if self.coef_ is None:
                return 0.0
            pred = self.predict(X)
            return 1.0 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    # Create a mock module for online_linear_model
    mock_online_linear_model = MagicMock()
    mock_online_linear_model.OnlineLinearModel = MockOnlineLinearModel
    sys.modules['ondil.estimators.online_linear_model'] = mock_online_linear_model
    
    return MockOnlineLinearModel

def test_real_adapter():
    """Test the real adapter with mocked dependencies."""
    
    print("🧪 Setting up mocks for ondil dependencies...")
    MockOnlineLinearModel = create_mock_ondil_dependencies()
    
    try:
        # Now try to import the real adapter
        from ondil.estimators.sktime_adapter import OnlineLinearModelSktime
        print("✓ OnlineLinearModelSktime imported successfully with mocks")
        
        # Test basic instantiation
        adapter = OnlineLinearModelSktime()
        print("✓ OnlineLinearModelSktime instantiated successfully")
        
        # Test parameter setting
        adapter2 = OnlineLinearModelSktime(
            forget=0.1,
            scale_inputs=False,
            fit_intercept=False,
            method="ols",
            ic="aic"
        )
        print("✓ Parameter setting works correctly")
        
        # Test 2D data
        X_2d = np.random.random((50, 5))
        y_2d = np.random.random(50)
        
        adapter.fit(X_2d, y_2d)
        pred_2d = adapter.predict(X_2d)
        
        assert pred_2d.shape == (50,)
        print("✓ 2D data fitting and prediction works")
        
        # Test 3D data
        X_3d = np.random.random((30, 4, 3))
        y_3d = np.random.random(30)
        
        adapter_3d = OnlineLinearModelSktime(fit_intercept=True)
        adapter_3d.fit(X_3d, y_3d)
        pred_3d = adapter_3d.predict(X_3d)
        
        assert pred_3d.shape == (30,)
        print("✓ 3D data fitting and prediction works")
        
        # Test update functionality
        X_new = np.random.random((10, 5))
        y_new = np.random.random(10)
        adapter.update(X_new, y_new)
        print("✓ Update functionality works")
        
        # Test score
        score = adapter.score(X_2d, y_2d)
        assert isinstance(score, float)
        print("✓ Score method works")
        
        # Test properties
        assert hasattr(adapter, 'coef_')
        assert hasattr(adapter, 'coef_path_')
        print("✓ Coefficient properties accessible")
        
        # Test fitted params
        fitted_params = adapter.get_fitted_params()
        assert isinstance(fitted_params, dict)
        print("✓ get_fitted_params works")
        
        # Test test params
        test_params = OnlineLinearModelSktime.get_test_params()
        assert isinstance(test_params, list)
        assert len(test_params) > 0
        print("✓ get_test_params works")
        
        print("\n🎉 All real adapter tests passed with mocks!")
        return True
        
    except Exception as e:
        print(f"❌ Real adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing sktime adapter with mocked dependencies")
    print("=" * 60)
    
    success = test_real_adapter()
    
    if success:
        print("\n🎉 All tests passed! The sktime adapter is working correctly.")
        print("Note: This was tested with mocked ondil dependencies.")
        print("The adapter should work with the real ondil package once numba is available.")
    else:
        print("\n❌ Tests failed")