import sys
sys.path.insert(1, '../azlearn')

from neural_network.MLP import MLPRegressor as MyMLPRegressor
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(20, 1)  # 100 samples, 1 feature
y = 2 * X.squeeze() + 3 + np.random.randn(20)  # true relationship: y = 2*X + 3 + noise

print("comparing linear regression models")

myModel = MyMLPRegressor()
sklModel = SklearnMLPRegressor()

myModel.fit(X, y.reshape(-1, 1))  # Reshape y to have two dimensions
sklModel.fit(X, y)

y_pred = myModel.predict(X)
y_pred2 = sklModel.predict(X)

for i in range(len(y)):
    print(f"Actual: {y[i]}, our model: {y_pred[i]}, sklearn model: {y_pred2[i]}")
