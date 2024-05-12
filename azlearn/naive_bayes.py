import sys

sys.path.insert(1, '.')

import numpy as np
from base import BaseEstimator, ClassifierMixin

class _BaseNB(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = None
        self.class_prior_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_prior_ = np.zeros(len(self.classes_))
        for i, c in enumerate(self.classes_):
            self.class_prior_[i] = np.mean(y == c)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        # Dummy implementation, to be overridden by subclasses
        n_samples, _ = X.shape
        return np.ones((n_samples, len(self.classes_))) / len(self.classes_)

class GaussianNaiveBayes(_BaseNB):
    def __init__(self):
        super().__init__()
        self.theta_ = None  # mean of each feature per class
        self.sigma_ = None  # variance of each feature per class

    def fit(self, X, y):
        super().fit(X, y)
        self.theta_ = np.zeros((len(self.classes_), X.shape[1]))
        self.sigma_ = np.zeros((len(self.classes_), X.shape[1]))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i] = np.mean(X_c, axis=0)
            self.sigma_[i] = np.var(X_c, axis=0)

    def _gaussian_pdf(self, X, mean, var):
        return 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (X - mean) ** 2 / var)

    def predict_proba(self, X):
        n_samples, n_features = X.shape
        probabilities = np.zeros((n_samples, len(self.classes_)))
        for i, c in enumerate(self.classes_):
            class_prob = self.class_prior_[i]
            for j in range(n_features):
                class_prob *= self._gaussian_pdf(X[:, j], self.theta_[i, j], self.sigma_[i, j])
            probabilities[:, i] = class_prob
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)
