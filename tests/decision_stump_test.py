import sys
sys.path.insert(1, '../azlearn')

from tree_model import DecisionStump
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Generate synthetic data
np.random.seed(35)
X = np.random.rand(20, 2)  # 20 samples, 2 features
y = np.random.randint(2, size=20)  # binary labels

print("Comparing decision stump and decision tree models")

# Train decision stump model
stump_model = DecisionStump.DecisionStump()
stump_model.fit(X, y)

# Train decision tree model (for comparison)
tree_model = DecisionTreeClassifier(random_state=35)
tree_model.fit(X, y)

# Make predictions
y_pred = stump_model.predict(X)
y_pred2 = tree_model.predict(X)

# Print predictions
for i in range(len(y)):
    print(f"Actual: {y[i]}, Our model: {y_pred[i]}, scikit-learn model: {y_pred2[i]}")

print("\nPredicted probabilities comparison")
probas = stump_model.predict_proba(X)
probas2 = tree_model.predict_proba(X)
for i in range(len(y)):
    print(f"Our model: {probas[i]}, scikit-learn model: {probas2[i]}")