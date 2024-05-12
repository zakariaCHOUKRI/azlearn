from base import Transformer,TransformerMixin
import numpy as np

class _BaseEncoder(Transformer,TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


    def inverse_transform(self, X):
        return X


class OneHotEncoder(_BaseEncoder):
    def __init__(self):
        self.unique_values_ = None

    def fit(self, X):
        self.unique_values_ = np.unique(X)
        return self

    def transform(self, X):
        encoded = np.zeros((len(X), len(self.unique_values_)), dtype=int)
        for i, val in enumerate(X):
            idx = np.where(self.unique_values_ == val)[0][0]
            encoded[i, idx] = 1
        return encoded


class LabelEncoder(_BaseEncoder):
    def __init__(self):
        self.unique_values_ = None

    def fit(self, X):
        self.unique_values_ = np.unique(X)
        return self

    def transform(self, X):
        encoded = np.zeros(len(X), dtype=int)
        for i, val in enumerate(X):
            idx = np.where(self.unique_values_ == val)[0][0]
            encoded[i] = idx
        return encoded
