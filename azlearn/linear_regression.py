# azlearn/linear_regression.py

from azlearn.base import BaseRegressor
import numpy as np

class LinearRegression(BaseRegressor):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit linear regression model.

        Parameters:
        - X: Input features (numpy array or pandas DataFrame)
        - y: Target values (numpy array or pandas Series)
        """
        # Add a column of ones for intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Calculate coefficients using normal equation
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
        # Set coefficients and intercept
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Input features (numpy array or pandas DataFrame)

        Returns:
        - Predicted target values (numpy array)
        """
        # Add a column of ones for intercept term
        X = np.c_[np.ones(X.shape[0]), X]

        # Calculate predictions
        return X.dot(self.theta)
