import sys
sys.path.insert(1, '../azlearn')

from tree_model import DecisionTreeRegressor as TreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

# Train decision tree model
tree_model = TreeRegressor.DecisionTreeRegressor()
tree_model.fit(X, y)

dtree_model = DecisionTreeRegressor(random_state=42)
dtree_model.fit(X, y)

# Make predictions
y_pred = tree_model.predict(X)
y_pred2 = dtree_model.predict(X)

# Print predictions
for i in range(min(len(y), 20)):
    print(f"Actual: {y[i]}, Our model: {y_pred[i]}, scikit-learn model: {y_pred2[i]}")
