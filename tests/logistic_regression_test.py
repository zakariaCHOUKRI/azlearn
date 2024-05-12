import sys
sys.path.insert(1, '../azlearn')

from linear_model.LogisticRegression import LogisticRegression as MyLogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import numpy as np

# Generate synthetic data
np.random.seed(35)
X = np.random.rand(20, 2)  # 20 samples, 2 features
y = np.random.randint(2, size=20)  # binary labels

print("Comparing logistic regression models")

myModel = MyLogisticRegression()
sklModel = SklearnLogisticRegression()

myModel.fit(X, y)
sklModel.fit(X, y)

y_pred = myModel.predict(X)
y_pred2 = sklModel.predict(X)

for i in range(len(y)):
    print(f"Actual: {y[i]}, Our model: {y_pred[i]}, scikit-learn model: {y_pred2[i]}")

print("\nPredicted probabilities comparison")
probas = myModel.predict_proba(X)
probas2 = sklModel.predict_proba(X)
for i in range(len(y)):
    print(f"Our model: {probas[i]}, scikit-learn model: {probas2[i]}")
