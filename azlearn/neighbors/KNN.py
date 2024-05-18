import sys
sys.path.insert(1, '..')

from azlearn.base import BaseClassifier, ClassifierMixin
from collections import Counter
import numpy as np

class KNeighborsClassifier(BaseClassifier, ClassifierMixin):
    def __init__(self, n_neighbors=5, algorithm='auto'):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm  # For simplicity, we will not implement different algorithms here
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the k-nearest neighbors classifier from the training dataset.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        self : object
            Returns self.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _compute_distances(self, X):
        """
        Compute the distance from each sample in X to each sample in the training data.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        distances : array-like of shape (n_samples, n_train_samples)
            The distances between each input sample and each training sample.
        """
        distances = np.sqrt(((X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2).sum(axis=2))
        return distances

    def _predict_sample(self, x):
        """
        Predict the class label for a single sample using k-nearest neighbors.

        Parameters:
        x : array-like of shape (n_features,)
            The input sample.

        Returns:
        int
            The predicted class label.
        """
        distances = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
        nearest_neighbor_ids = distances.argsort()[:self.n_neighbors]
        nearest_neighbor_labels = self.y_train[nearest_neighbor_ids]
        most_common = Counter(nearest_neighbor_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        X = np.array(X)
        return np.array([self._predict_sample(x) for x in X])

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns:
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
