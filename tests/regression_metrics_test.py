from sklearn.metrics import mean_squared_error as sk_mean_squared_error
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import r2_score as sk_r2_score

import sys
sys.path.insert(1, '../azlearn')
from metrics.metrics import mean_squared_error,mean_absolute_error,r2_score

import numpy as np

# Generate sample data
np.random.seed(42)
y_true_regression = np.random.rand(100)
y_pred_regression = np.random.rand(100)

# Test regression metrics
print("Regression metrics:")

# Custom implementation
print("Custom implementation:")
print("Mean Squared Error:", mean_squared_error(y_true_regression, y_pred_regression))
print("Mean Absolute Error:", mean_absolute_error(y_true_regression, y_pred_regression))
print("R-squared (Coefficient of Determination):", r2_score(y_true_regression, y_pred_regression))

print("")
# Sklearn implementation
print("Sklearn implementation:")
print("Mean Squared Error:", sk_mean_squared_error(y_true_regression, y_pred_regression))
print("Mean Absolute Error:", sk_mean_absolute_error(y_true_regression, y_pred_regression))
print("R-squared (Coefficient of Determination):", sk_r2_score(y_true_regression, y_pred_regression))
