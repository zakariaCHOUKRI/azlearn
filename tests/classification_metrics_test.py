from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import accuracy_score as sk_accuracy_score

import sys
sys.path.insert(1, '../azlearn')
from metrics.metrics import precision_score, recall_score, f1score, accuracy_score



import numpy as np

np.random.seed(42)
y_true = np.random.choice(['spam', 'ham'], size=100)
y_pred = np.random.choice(['spam', 'ham'], size=100)

# Generate sample data


# Test binary classification
print("Binary classification:")
y_true_binary = np.where(y_true == 'spam', 1, 0)
y_pred_binary = np.where(y_pred == 'spam', 1, 0)

# Custom implementation
print("Custom implementation:")
print("Precision:", precision_score(y_true_binary, y_pred_binary))
print("Recall:", recall_score(y_true_binary, y_pred_binary))
print("F1 Score:", f1score(y_true_binary, y_pred_binary))
print("Accuracy:", accuracy_score(y_true_binary, y_pred_binary))

print("")
# Sklearn implementation
print("Sklearn implementation:")
print("Precision:", sk_precision_score(y_true_binary, y_pred_binary))
print("Recall:", sk_recall_score(y_true_binary, y_pred_binary))
print("F1 Score:", sk_f1_score(y_true_binary, y_pred_binary))
print("Accuracy:", sk_accuracy_score(y_true_binary, y_pred_binary))


np.random.seed(42)
y_true = np.random.choice(['spam', 'ham', 'other'], size=100)
y_pred = np.random.choice(['spam', 'ham', 'other'], size=100)

# Test multiclass classification
print("\nMulticlass classification:")

# Custom implementation
print("Custom (micro)")
print("Custom implementation:")
print("Precision (micro):", precision_score(y_true, y_pred, average='micro'))
print("Recall (micro):", recall_score(y_true, y_pred, average='micro'))
print("F1 Score (micro):", f1score(y_true, y_pred, average='micro'))
print("Accuracy (micro):", accuracy_score(y_true, y_pred))

print("")
print("Custom (macro)")

print("Precision (macro):", precision_score(y_true, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_true, y_pred, average='macro'))
print("F1 Score (macro):", f1score(y_true, y_pred, average='macro'))
print("Accuracy (macro):", accuracy_score(y_true, y_pred))

print("Custom (weighted)")
print("")
print("Precision (weighted):", precision_score(y_true, y_pred, average='weighted'))
print("Recall (weighted):", recall_score(y_true, y_pred, average='weighted'))
print("F1 Score (weighted):", f1score(y_true, y_pred, average='weighted'))
print("Accuracy (weighted):", accuracy_score(y_true, y_pred))

print("")
# Sklearn implementation
print("Sklearn implementation:")
print("SKlearn (micro)")
print("Precision (micro):", sk_precision_score(y_true, y_pred, average='micro'))
print("Recall (micro):", sk_recall_score(y_true, y_pred, average='micro'))
print("F1 Score (micro):", sk_f1_score(y_true, y_pred, average='micro'))
print("Accuracy (micro):", sk_accuracy_score(y_true, y_pred))

print("")
print("Sklearn (macro):")
print("Precision (macro):", sk_precision_score(y_true, y_pred, average='macro'))
print("Recall (macro):", sk_recall_score(y_true, y_pred, average='macro'))
print("F1 Score (macro):", sk_f1_score(y_true, y_pred, average='macro'))
print("Accuracy (macro):", sk_accuracy_score(y_true, y_pred))

print("")
print("Sklearn (weighted):")
print("Precision (weighted):", sk_precision_score(y_true, y_pred, average='weighted'))
print("Recall (weighted):", sk_recall_score(y_true, y_pred, average='weighted'))
print("F1 Score (weighted):", sk_f1_score(y_true, y_pred, average='weighted'))
print("Accuracy (weighted):", sk_accuracy_score(y_true, y_pred))
