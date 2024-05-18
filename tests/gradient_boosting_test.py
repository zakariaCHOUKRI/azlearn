import sys
sys.path.insert(1, '../azlearn')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from azlearn.ensemble.GradientBoost import GradientBoostingClassifier

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print("Comparing azlearn GradientBoostingClassifier and sklearn GradientBoostingClassifier")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train azlearn GradientBoostingClassifier model
az_gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
az_gb_model.fit(X_train, y_train)

# Train sklearn GradientBoostingClassifier model (for comparison)
sklearn_gb_model = SklearnGradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
sklearn_gb_model.fit(X_train, y_train)

# Make predictions
y_pred_az = az_gb_model.predict(X_test)
y_pred_sklearn = sklearn_gb_model.predict(X_test)

# Print predictions comparison
print("Predictions comparison (first 20 samples)")
for i in range(min(len(y_pred_az), 20)):
    print(f"Actual: {y_test[i]}, azlearn: {y_pred_az[i]}, sklearn: {y_pred_sklearn[i]}")

# Evaluate models on the test set
accuracy_az = accuracy_score(y_test, y_pred_az)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print("\nAccuracy comparison")
print(f"azlearn GradientBoostingClassifier accuracy: {accuracy_az:.4f}")
print(f"sklearn GradientBoostingClassifier accuracy: {accuracy_sklearn:.4f}")


# Predicted probabilities comparison
probas_az = az_gb_model.predict_proba(X_test)
probas_sklearn = sklearn_gb_model.predict_proba(X_test)
print("\nPredicted probabilities comparison (first 5 samples)")
for i in range(min(len(probas_az), 5)):
    print(f"azlearn: {probas_az[i]}, sklearn: {probas_sklearn[i]}")