import sys
sys.path.insert(1, '..')

from base import Predictor,RegressorMixin
import numpy as np

class LinearRegression(Predictor, RegressorMixin):
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """
        Fit linear regression model to the training data using Ordinary Least Squares (OLS).

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        # Add a column of ones to X for the intercept term
        X_with_intercept = np.column_stack((np.ones_like(y), X))

        # Compute the coefficients using OLS formula: beta = (X^T * X)^(-1) * X^T * y
        self.coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

        # The first coefficient is the intercept, and the rest are the coefficients for features
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        """
        Predict using the linear regression model.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Samples.

        Returns:
        array-like of shape (n_samples,)
            Predicted values.
        """
        # Add a column of ones to X for the intercept term
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

        # Use the computed coefficients to make predictions
        predictions = X_with_intercept @ np.concatenate(([self.intercept], self.coefficients))

        return predictions