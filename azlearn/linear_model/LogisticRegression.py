import sys

sys.path.insert(1, '..')

from azlearn.base import BaseClassifier
import numpy as np


class LogisticRegression(BaseClassifier):
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z : array-like
            Input to the sigmoid function.

        Returns:
        array-like
            Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit logistic regression model to the training data using gradient descent.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_iterations):
            # Calculate the predicted probabilities
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            # Compute gradients
            dw = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict using the logistic regression model.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Samples.

        Returns:
        array-like of shape (n_samples,)
            Predicted class labels.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Calculate predicted probabilities
        z = np.dot(X, self.weights) + self.bias
        y_pred_proba = self._sigmoid(z)

        # Convert probabilities to class labels
        y_pred = np.where(y_pred_proba >= 0.5, 1, 0)

        return y_pred

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
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Calculate predicted probabilities
        z = np.dot(X, self.weights) + self.bias
        y_pred_proba = self._sigmoid(z)

        # Create array of shape (n_samples, 2) with probabilities for class 0 and class 1
        proba = np.column_stack((1 - y_pred_proba, y_pred_proba))

        return proba
