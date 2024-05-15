import sys
sys.path.insert(1, '../azlearn')

from linear_model.LogisticRegression import LogisticRegression as MyLogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.datasets import load_breast_cancer
import numpy as np

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Comparing logistic regression models")

myModel = MyLogisticRegression()
sklModel = SklearnLogisticRegression()

myModel.fit(X, y)
sklModel.fit(X, y)

y_pred = myModel.predict(X)
y_pred2 = sklModel.predict(X)

for i in range(min(len(y), 20)):
    print(f"Actual: {y[i]}, Our model: {y_pred[i]}, scikit-learn model: {y_pred2[i]}")

print("\nPredicted probabilities comparison")
probas = myModel.predict_proba(X)
probas2 = sklModel.predict_proba(X)
for i in range(min(len(y), 20)):
    print(f"Our model: {probas[i]}, scikit-learn model: {probas2[i]}")
