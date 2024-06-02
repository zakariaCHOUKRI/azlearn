import numpy as np
from base import Transformer,TransformerMixin
import numpy as np



class SimpleImputer(Transformer,TransformerMixin):
    def __init__(self, missing_values=np.nan, strategy='mean'):
        super().__init__()
        self.missing_values = missing_values
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.statistics_ = self._most_frequent(X)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'.")
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise ValueError("SimpleImputer has not been fitted yet.")

        filled_X = X.copy()
        for i, col in enumerate(filled_X.T):
            missing_mask = np.isnan(col)
            if missing_mask.any():
                filled_X[missing_mask, i] = self.statistics_[i]
        return filled_X

    def _most_frequent(self, X):
        most_frequent_vals = []
        for col in X.T:
            col_values = col[~np.isnan(col)]
            if len(col_values) == 0:
                # If all values are missing, fill with 0
                most_frequent_vals.append(0)
            else:
                unique, counts = np.unique(col_values, return_counts=True)
                most_frequent_val = unique[np.argmax(counts)]
                most_frequent_vals.append(most_frequent_val)
        return np.array(most_frequent_vals)