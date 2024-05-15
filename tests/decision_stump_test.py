import sys
sys.path.insert(1, '../azlearn')


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tree_model import DecisionStump
from sklearn.tree import DecisionTreeClassifier

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print("Comparing decision stump and decision tree models")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision stump model
stump_model = DecisionStump.DecisionStump()
stump_model.fit(X_train, y_train)

# Train decision tree model (for comparison)
tree_model = DecisionTreeClassifier(random_state=35)
tree_model.fit(X_train, y_train)

# Make predictions
y_pred = stump_model.predict(X_test)
y_pred2 = tree_model.predict(X_test)

# Print predictions
print("Predictions comparison")
for i in range(min(len(y_pred), 20)):
    print(f"Actual: {y_test[i]}, Our model (Decision Stump): {y_pred[i]}, scikit-learn model (Decision Tree): {y_pred2[i]}")


print("\nPredicted probabilities comparison")
probas = stump_model.predict_proba(X)
probas2 = tree_model.predict_proba(X)
for i in range(min(len(y), 20)):
    print(f"Our model: {probas[i]}, scikit-learn model: {probas2[i]}")


# Evaluate models on the test set
accuracy_stump = stump_model.score(X_test, y_test)
accuracy_tree = tree_model.score(X_test, y_test)

print("\nAccuracy comparison")
print("Decision Stump accuracy:", accuracy_stump)
print("Decision Tree accuracy:", accuracy_tree)
