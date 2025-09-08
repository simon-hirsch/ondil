#!/usr/bin/env python3
"""
Example usage of the OnlineLinearModelSktime adapter.

This script demonstrates how to use the sktime adapter for OnlineLinearModel
with both 2D and 3D data formats.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def example_basic_usage():
    """Demonstrate basic usage with 2D data."""
    print("=" * 60)
    print("Example 1: Basic Usage with 2D Data")
    print("=" * 60)
    
    try:
        from ondil.estimators import OnlineLinearModelSktime
        
        # Generate sample data
        X, y = make_regression(
            n_samples=200, 
            n_features=5, 
            noise=0.1, 
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Create and fit adapter
        adapter = OnlineLinearModelSktime(
            method="ols",
            fit_intercept=True,
            scale_inputs=True
        )
        
        print("\nFitting model...")
        adapter.fit(X_train, y_train)
        
        # Make predictions
        y_pred = adapter.predict(X_test)
        score = adapter.score(X_test, y_test)
        
        print(f"R² score: {score:.4f}")
        print(f"Model coefficients shape: {adapter.coef_.shape}")
        print(f"Coefficients: {adapter.coef_}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import adapter: {e}")
        print("This is likely due to missing dependencies (numba)")
        return False
    except Exception as e:
        print(f"❌ Error in basic usage example: {e}")
        return False

def example_3d_data():
    """Demonstrate usage with 3D time series data."""
    print("\n" + "=" * 60)
    print("Example 2: 3D Time Series Data")
    print("=" * 60)
    
    try:
        from ondil.estimators import OnlineLinearModelSktime
        
        # Generate 3D time series data
        n_series = 100
        n_timepoints = 10
        n_features = 3
        
        # Create synthetic time series data
        X_3d = np.random.random((n_series, n_timepoints, n_features))
        
        # Create targets based on flattened features
        X_flat = X_3d.reshape(n_series, -1)
        true_coef = np.random.random(n_timepoints * n_features) * 0.5
        y_3d = X_flat @ true_coef + np.random.normal(0, 0.1, n_series)
        
        print(f"3D data shape: {X_3d.shape}")
        print(f"Targets shape: {y_3d.shape}")
        print(f"Flattened features: {n_timepoints * n_features}")
        
        # Split data
        n_train = int(0.7 * n_series)
        X_train_3d = X_3d[:n_train]
        y_train_3d = y_3d[:n_train]
        X_test_3d = X_3d[n_train:]
        y_test_3d = y_3d[n_train:]
        
        # Create and fit adapter
        adapter = OnlineLinearModelSktime(
            method="lasso",
            fit_intercept=True,
            ic="aic"
        )
        
        print("\nFitting model with 3D data...")
        adapter.fit(X_train_3d, y_train_3d)
        
        # Make predictions
        y_pred_3d = adapter.predict(X_test_3d)
        score = adapter.score(X_test_3d, y_test_3d)
        
        print(f"R² score: {score:.4f}")
        print(f"Predicted coefficients shape: {adapter.coef_.shape}")
        print(f"Non-zero coefficients: {np.sum(np.abs(adapter.coef_) > 1e-6)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import adapter: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in 3D data example: {e}")
        return False

def example_online_learning():
    """Demonstrate online learning capabilities."""
    print("\n" + "=" * 60)
    print("Example 3: Online Learning")
    print("=" * 60)
    
    try:
        from ondil.estimators import OnlineLinearModelSktime
        
        # Generate initial training data
        X_initial, y_initial = make_regression(
            n_samples=100, 
            n_features=3, 
            noise=0.1, 
            random_state=42
        )
        
        print(f"Initial training data: {X_initial.shape}")
        
        # Create adapter with forgetting factor
        adapter = OnlineLinearModelSktime(
            method="ols",
            forget=0.05,  # Small forgetting factor
            fit_intercept=True
        )
        
        # Initial fit
        print("\nInitial fit...")
        adapter.fit(X_initial, y_initial)
        initial_coef = adapter.coef_.copy()
        print(f"Initial coefficients: {initial_coef}")
        
        # Simulate streaming data updates
        print("\nStreaming updates:")
        coefficients_history = [initial_coef]
        
        for batch in range(5):
            # Generate new batch
            X_batch, y_batch = make_regression(
                n_samples=20, 
                n_features=3, 
                noise=0.1, 
                random_state=42 + batch
            )
            
            # Update model
            adapter.update(X_batch, y_batch)
            current_coef = adapter.coef_.copy()
            coefficients_history.append(current_coef)
            
            print(f"  Batch {batch + 1}: {current_coef}")
        
        # Show coefficient evolution
        print(f"\nCoefficient change from initial:")
        final_change = coefficients_history[-1] - coefficients_history[0]
        print(f"  Change: {final_change}")
        print(f"  Max change: {np.max(np.abs(final_change)):.4f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import adapter: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in online learning example: {e}")
        return False

def example_method_comparison():
    """Compare different estimation methods."""
    print("\n" + "=" * 60)
    print("Example 4: Method Comparison")
    print("=" * 60)
    
    try:
        from ondil.estimators import OnlineLinearModelSktime
        
        # Generate data with some irrelevant features
        X, y = make_regression(
            n_samples=150,
            n_features=10,
            n_informative=5,  # Only 5 features are informative
            noise=0.1,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        methods = ["ols", "lasso", "elasticnet"]
        results = {}
        
        print(f"Data shape: {X_train.shape}")
        print(f"Testing methods: {methods}")
        print()
        
        for method in methods:
            try:
                adapter = OnlineLinearModelSktime(
                    method=method,
                    fit_intercept=True,
                    ic="bic"
                )
                
                adapter.fit(X_train, y_train)
                score = adapter.score(X_test, y_test)
                n_nonzero = np.sum(np.abs(adapter.coef_) > 1e-6)
                
                results[method] = {
                    'score': score,
                    'n_features': n_nonzero,
                    'coef_': adapter.coef_
                }
                
                print(f"{method.upper():>10}: R² = {score:.4f}, Features = {n_nonzero}")
                
            except Exception as e:
                print(f"{method.upper():>10}: Failed ({e})")
        
        # Show feature selection differences
        if 'lasso' in results and 'ols' in results:
            print(f"\nFeature selection comparison:")
            print(f"  OLS uses all {results['ols']['n_features']} features")
            print(f"  Lasso selected {results['lasso']['n_features']} features")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import adapter: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in method comparison: {e}")
        return False

def main():
    """Run all examples."""
    print("OnlineLinearModelSktime Adapter Examples")
    print("========================================")
    
    examples = [
        example_basic_usage,
        example_3d_data,
        example_online_learning,
        example_method_comparison,
    ]
    
    successes = 0
    
    for example in examples:
        try:
            if example():
                successes += 1
        except Exception as e:
            print(f"❌ Example failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Summary: {successes}/{len(examples)} examples completed successfully")
    
    if successes == 0:
        print("\n❌ All examples failed. This is likely due to missing dependencies.")
        print("Make sure you have installed ondil with numba support:")
        print("  pip install ondil")
    elif successes < len(examples):
        print(f"\n⚠️  Some examples failed. Check the error messages above.")
    else:
        print("\n🎉 All examples completed successfully!")
        print("\nThe OnlineLinearModelSktime adapter is working correctly.")

if __name__ == "__main__":
    main()