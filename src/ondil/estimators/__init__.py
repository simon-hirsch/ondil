from .online_gamlss import OnlineDistributionalRegression
from .online_lasso import OnlineLasso
from .online_linear_model import OnlineLinearModel

# Try to import sktime adapter conditionally
try:
    from .sktime_adapter import OnlineLinearModelSktime
    __all__ = ["OnlineDistributionalRegression", "OnlineLasso", "OnlineLinearModel", "OnlineLinearModelSktime"]
except ImportError:
    # sktime not available, don't expose the adapter
    __all__ = ["OnlineDistributionalRegression", "OnlineLasso", "OnlineLinearModel"]
