import sys
sys.path.insert(1, '../azlearn')  # Modify the path accordingly

from naive_bayes import GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
import numpy as np

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Train Gaussian Naive Bayes model
gnb_model = GaussianNaiveBayes()
gnb_model.fit(X, y)

sk_gnb_model = GaussianNB()
sk_gnb_model.fit(X, y)

# Make predictions
y_pred = gnb_model.predict(X)
y_pred2 = sk_gnb_model.predict(X)

# Print predictions
for i in range(min(len(y), 20)):
    print(f"Actual: {y[i]}, Our model: {y_pred[i]}, scikit-learn model: {y_pred2[i]}")

print("\nPredicted probabilities comparison")
probas = gnb_model.predict_proba(X)
probas2 = sk_gnb_model.predict_proba(X)
for i in range(min(len(y), 20)):
    print(f"Our model: {probas[i]}, scikit-learn model: {probas2[i]}")
