import sys
sys.path.insert(1, '../azlearn')

from tree_model import DecisionTreeRegressor as TreeRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(20, 2)  # 20 samples, 2 features
y = np.random.rand(20)  # continuous target values

# Train decision tree model
tree_model = TreeRegressor.DecisionTreeRegressor()
tree_model.fit(X, y)

dtree_model = DecisionTreeRegressor(random_state=42)
dtree_model.fit(X, y)

# Make predictions
y_pred = tree_model.predict(X)
y_pred2 = dtree_model.predict(X)

# Print predictions
for i in range(len(y)):
    print(f"Actual: {y[i]}, Our model: {y_pred[i]}, scikit-learn model: {y_pred2[i]}")
