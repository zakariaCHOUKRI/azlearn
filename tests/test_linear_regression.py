# tests/test_linear_regression.py

import sys
sys.path.insert(1, '../azlearn')

import numpy as np
import pytest
from azlearn.linear_regression import LinearRegression

# Sample data for testing
X_train = np.array([[1], [2], [3], [4]])
y_train = np.array([2, 4, 5, 4])
X_test = np.array([[5], [6]])

def test_linear_regression_fit_predict():
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Check coefficients and intercept
    assert np.allclose(model.coef_, [0.6])  # Expected coefficient
    assert np.isclose(model.intercept_, 2.2)  # Expected intercept

    # Make predictions
    predictions = model.predict(X_test)

    # Check predictions
    assert np.allclose(predictions, [4.6, 5.2])  # Expected predictions

if __name__ == "__main__":
    pytest.main([__file__])
