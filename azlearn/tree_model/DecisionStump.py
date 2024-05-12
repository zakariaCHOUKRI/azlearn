import sys
sys.path.insert(1, '..')

from azlearn.base import BaseClassifier, ClassifierMixin
import numpy as np

class DecisionStump(BaseClassifier, ClassifierMixin):
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.prediction = None

    def fit(self, X, y):
        """
        Fit decision stump to the training data.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        best_error = float('inf')

        # Iterate over all features and thresholds to find the best split
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                prediction = np.mean(y[X[:, feature_index] <= threshold])
                error = np.sum(y != (X[:, feature_index] <= threshold))
                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.prediction = prediction

    def predict(self, X):
        """
        Predict using the decision stump model.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Samples.

        Returns:
        array-like of shape (n_samples,)
            Predicted class labels.
        """
        if self.feature_index is None or self.threshold is None or self.prediction is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        x = np.where(X[:, self.feature_index] <= self.threshold, self.prediction, 1 - self.prediction)
        return np.round(x).astype(int)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Samples.

        Returns:
        array-like of shape (n_samples, 2)
            Class probabilities.
        """
        if self.feature_index is None or self.threshold is None or self.prediction is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        proba = np.zeros((X.shape[0], 2))
        proba[:, 1] = np.where(X[:, self.feature_index] <= self.threshold, self.prediction, 1 - self.prediction)
        proba[:, 0] = 1 - proba[:, 1]
        return proba
