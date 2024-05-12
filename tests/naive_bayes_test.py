import sys
sys.path.insert(1, '../azlearn')  # Modify the path accordingly

from naive_bayes import GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(20, 2)  # 20 samples, 2 features
y = np.random.randint(2, size=20)  # binary labels

# Train Gaussian Naive Bayes model
gnb_model = GaussianNaiveBayes()
gnb_model.fit(X, y)

sk_gnb_model = GaussianNB()
sk_gnb_model.fit(X, y)

# Make predictions
y_pred = gnb_model.predict(X)
y_pred2 = sk_gnb_model.predict(X)

# Print predictions
for i in range(len(y)):
    print(f"Actual: {y[i]}, Our model: {y_pred[i]}, scikit-learn model: {y_pred2[i]}")

print("\nPredicted probabilities comparison")
probas = gnb_model.predict_proba(X)
probas2 = sk_gnb_model.predict_proba(X)
for i in range(len(y)):
    print(f"Our model: {probas[i]}, scikit-learn model: {probas2[i]}")
