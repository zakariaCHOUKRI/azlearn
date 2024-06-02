import numpy as np
from scipy.stats import chi2_contingency

import sys

sys.path.insert(1, '../azlearn')
from azlearn.base import BaseEstimator, Transformer


class SelectKBestFeatures(BaseEstimator, Transformer):
    def __init__(self, k=10):
        self.k = k
        self.scores_ = None
        self.selected_indices_ = None

    def fit(self, X, y):
        num_features = X.shape[1]
        scores = np.zeros(num_features)

        for i in range(num_features):
            scores[i] = self._mutual_info_score(X[:, i], y)

        # Get indices of top k features
        self.selected_indices_ = np.argsort(scores)[::-1][:self.k]
        self.scores_ = scores[self.selected_indices_]
        return self

    def transform(self, X):
        if self.selected_indices_ is None:
            raise ValueError("SelectKBestFeatures has not been fitted yet.")
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        raise NotImplementedError("inverse_transform method not implemented")

    def _mutual_info_score(self, x, y):
        contingency_table = np.histogram2d(x, y, bins=(len(np.unique(x)), len(np.unique(y))))[0]
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = np.sum(contingency_table)
        return 0.5 * np.log(1 + chi2 / n)


class VarianceThreshold(BaseEstimator, Transformer):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self._variances = None

    def fit(self, X, y=None):
        self._variances = np.var(X, axis=0)
        return self

    def transform(self, X):
        if self._variances is None:
            raise ValueError("VarianceThreshold has not been fitted yet.")

        mask = self._variances > self.threshold
        return X[:, mask]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        raise NotImplementedError("inverse_transform method not implemented")
