import sys
sys.path.insert(1, '../azlearn')

from tree_model import DecisionTreeClassifier as TreeClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(20, 2)  # 20 samples, 2 features
y = np.random.randint(2, size=20)  # binary labels

# Train decision tree model
tree_model = TreeClassifier.DecisionTreeClassifier()
tree_model.fit(X, y)

dtree_model = DecisionTreeClassifier(random_state=42)
dtree_model.fit(X, y)

# Make predictions
y_pred = tree_model.predict(X)
y_pred2 = dtree_model.predict(X)

# Print predictions
for i in range(len(y)):
    print(f"Actual: {y[i]}, Our model: {y_pred[i]}, scikit-learn model: {y_pred2[i]}")

print("\nPredicted probabilities comparison")
probas = tree_model.predict_proba(X)
probas2 = dtree_model.predict_proba(X)
for i in range(len(y)):
    print(f"Our model: {probas[i]}, scikit-learn model: {probas2[i]}")