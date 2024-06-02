import numpy as np
import sys
sys.path.insert(1, '../azlearn')
from ensemble.BaggingClassifier import BaggingClassifier
from model_selection.train_test_split import train_test_split
from tree_model.DecisionTreeClassifier import DecisionTreeClassifier
from datasets.synthetic_data import generate_synthetic_data


X, y = generate_synthetic_data(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Bagging Classifier
bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, max_samples=1.0, random_state=42)

# Train the model
bagging_clf.fit(X_train, y_train)

# Predict
y_pred = bagging_clf.predict(X_test)

# Evaluate
score = bagging_clf.score(X_test, y_test)
print(f"Accuracy: {score:.2f}")