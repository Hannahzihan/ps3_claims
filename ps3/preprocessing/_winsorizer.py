import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile, upper_quantile):
        self.lower_quantile=lower_quantile
        self.upper_quantile=upper_quantile
    def fit(self, X, y=None):
        # Calculate the lower and upper quantiles for each feature
        self.lower_quantile_ = np.percentile(X, self.lower_quantile * 100, axis=0)
        self.upper_quantile_ = np.percentile(X, self.upper_quantile * 100, axis=0)
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_quantile_, self.upper_quantile_)
        return X_clipped
