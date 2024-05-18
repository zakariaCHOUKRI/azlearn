import sys
sys.path.insert(1, '../azlearn')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from sklearn.metrics import accuracy_score
from azlearn.neighbors.KNN import KNeighborsClassifier

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print("Comparing azlearn KNeighborsClassifier and sklearn KNeighborsClassifier")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train azlearn KNeighborsClassifier model
az_knn_model = KNeighborsClassifier(n_neighbors=3)
az_knn_model.fit(X_train, y_train)

# Train sklearn KNeighborsClassifier model (for comparison)
sklearn_knn_model = SklearnKNeighborsClassifier(n_neighbors=3)
sklearn_knn_model.fit(X_train, y_train)

# Make predictions
y_pred_az = az_knn_model.predict(X_test)
y_pred_sklearn = sklearn_knn_model.predict(X_test)

# Print predictions comparison
print("Predictions comparison (first 20 samples)")
for i in range(min(len(y_pred_az), 20)):
    print(f"Actual: {y_test[i]}, azlearn: {y_pred_az[i]}, sklearn: {y_pred_sklearn[i]}")

# Evaluate models on the test set
accuracy_az = accuracy_score(y_test, y_pred_az)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print("\nAccuracy comparison")
print(f"azlearn KNeighborsClassifier accuracy: {accuracy_az:.4f}")
print(f"sklearn KNeighborsClassifier accuracy: {accuracy_sklearn:.4f}")

# Check if predictions match
if np.array_equal(y_pred_az, y_pred_sklearn):
    print("The predictions from azlearn and sklearn KNeighborsClassifier match!")
else:
    print("The predictions from azlearn and sklearn KNeighborsClassifier do not match.")

# Print the accuracy comparison
print(f"\nazlearn KNeighborsClassifier accuracy: {accuracy_az:.4f}")
print(f"sklearn KNeighborsClassifier accuracy: {accuracy_sklearn:.4f}")
