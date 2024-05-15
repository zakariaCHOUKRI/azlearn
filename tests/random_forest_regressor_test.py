import sys
sys.path.insert(1, '../azlearn')  # Modify the path accordingly

from azlearn.ensemble.Forest import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
from sklearn.datasets import load_diabetes
import numpy as np

# Load the Boston dataset for regression
boston_data = load_diabetes()
X_reg = boston_data.data
y_reg = boston_data.target

# Train RandomForestRegressor
our_rf_reg = RandomForestRegressor()
our_rf_reg.fit(X_reg, y_reg)

sk_rf_reg = SkRandomForestRegressor()
sk_rf_reg.fit(X_reg, y_reg)

# Make predictions for regression
y_pred_reg = our_rf_reg.predict(X_reg)
y_pred_reg_sk = sk_rf_reg.predict(X_reg)

# Print predictions for regression
print("\nPredictions for RandomForestRegressor:")
for i in range(min(len(y_reg), 20)):
    print(f"Actual: {y_reg[i]}, Our model: {y_pred_reg[i]:.2f}, scikit-learn model: {y_pred_reg_sk[i]:.2f}")
