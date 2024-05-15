import sys
sys.path.insert(1, '../azlearn')

from linear_model.LinearRegression import LinearRegression as MyLinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
data = load_diabetes()
X = data.data  # Using all features
y = data.target

print("comparing linear regression models")

myModel = MyLinearRegression()
sklModel = SklearnLinearRegression()

myModel.fit(X, y)
sklModel.fit(X, y)

y_pred = myModel.predict(X)
y_pred2 = sklModel.predict(X)

for i in range(min(len(y), 20)):
    print(f"Actual: {y[i]}, our model: {y_pred[i]}, sklearn model: {y_pred2[i]}")
