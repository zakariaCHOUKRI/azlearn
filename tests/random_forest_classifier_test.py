import sys
sys.path.insert(1, '../azlearn')  # Modify the path accordingly

from azlearn.ensemble.Forest import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.datasets import load_wine
import numpy as np

# Load the Iris dataset for classification
iris_data = load_wine()
X_cls = iris_data.data
y_cls = iris_data.target

# Train RandomForestClassifier
our_rf_cls = RandomForestClassifier()
our_rf_cls.fit(X_cls, y_cls)

sk_rf_cls = SkRandomForestClassifier()
sk_rf_cls.fit(X_cls, y_cls)

# Make predictions for classification
y_pred_cls = our_rf_cls.predict(X_cls)
y_pred_cls_sk = sk_rf_cls.predict(X_cls)

# Print predictions for classification
print("Predictions for RandomForestClassifier:")
for i in range(min(len(y_cls), 20)):
    print(f"Actual: {y_cls[i]}, Our model: {y_pred_cls[i]}, scikit-learn model: {y_pred_cls_sk[i]}")

print("\nPredicted probabilities comparison")
probas = our_rf_cls.predict_proba(X_cls)
probas2 = sk_rf_cls.predict_proba(X_cls)
for i in range(min(len(y_cls), 20)):
    print(f"Our model: {probas[i]}, scikit-learn model: {probas2[i]}")
