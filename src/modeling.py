import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class GrowthRegressor(BaseEstimator, RegressorMixin):
    """Predicts log(future value / current value), then converts back to euros.

    Forecasting the CHANGE is easier than forecasting the level, and it stops
    tree models from having to re-learn the scale of market values.
    """

    def __init__(self, estimator, base_col="log_current_value"):
        self.estimator = estimator
        self.base_col = base_col

    def fit(self, X, y):
        base = X[self.base_col].to_numpy()
        self.estimator_ = clone(self.estimator).fit(X, np.log1p(y) - base)
        return self

    def predict(self, X):
        return np.expm1(self.estimator_.predict(X) + X[self.base_col].to_numpy())
