import sys
sys.path.insert(1, '../azlearn')

from linear_model.LinearRegression import LinearRegression as MyLinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(20, 1)  # 100 samples, 1 feature
y = 2 * X.squeeze() + 3 + np.random.randn(20)  # true relationship: y = 2*X + 3 + noise

print("comparing linear regression models")

myModel = MyLinearRegression()
sklModel = SklearnLinearRegression()

myModel.fit(X, y)
sklModel.fit(X, y)

y_pred = myModel.predict(X)
y_pred2 = sklModel.predict(X)

for i in range(len(y)):
    print(f"Actual: {y[i]}, our model: {y_pred[i]}, sklearn model: {y_pred2[i]}")
