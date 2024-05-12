import sys
sys.path.insert(1, '../azlearn')

from neural_network.MLP import MLPClassifier as MyMLPClassifier
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
import numpy as np
from preprocessing import encoding

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(20, 2)  # 20 samples, 2 features
y = np.random.randint(2, size=20)  # binary labels

# One-hot encode the labels
encoder = encoding.OneHotEncoder()
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

print("Comparing MLP classifier models")

myModel = MyMLPClassifier(hidden_layer_sizes=(10,), activation='relu', learning_rate=0.01, max_iter=100)
sklModel = SklearnMLPClassifier(hidden_layer_sizes=(10,), activation='relu', learning_rate_init=0.01, max_iter=100)

myModel.fit(X, y)
sklModel.fit(X, y)

y_pred = myModel.predict(X)
y_pred2 = sklModel.predict(X)

print("Comparing predictions:")
for i in range(len(y)):
    print(f"Actual: {y[i]}, Our model: {y_pred[i]}, scikit-learn model: {y_pred2[i]}")

print("\nPredicted probabilities comparison")
probas = myModel.predict_proba(X)
probas2 = sklModel.predict_proba(X)
for i in range(len(y)):
    print(f"Our model: {probas[i]}, scikit-learn model: {probas2[i]}")
