import sys
sys.path.insert(1, '..')

from azlearn.base import BaseClassifier, ClassifierMixin, RegressorMixin
from azlearn.tree_model import DecisionTreeClassifier, DecisionTreeRegressor

import numpy as np
from abc import ABCMeta, abstractmethod
from collections import Counter

class BaseForest(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

class RandomForestClassifier(ForestClassifier):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.estimators = []
        self.n_classes = None  # New attribute to store the number of classes

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))  # Store the number of classes
        n_samples, n_features = X.shape
        max_features = self.max_features or int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            # Randomly select samples and features
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            feature_indices = np.random.choice(n_features, max_features, replace=False)

            X_subset = X[sample_indices][:, feature_indices]
            y_subset = y[sample_indices]

            # Fit decision tree on subset
            tree = DecisionTreeClassifier.DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, y_subset)
            self.estimators.append((tree, feature_indices))

    def predict_proba(self, X):
        if self.n_classes is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        predictions = np.zeros((X.shape[0], len(self.estimators), self.n_classes), dtype=float)
        for i, (tree, feature_indices) in enumerate(self.estimators):
            X_subset = X[:, feature_indices]
            predictions[:, i, :] = tree.predict_proba(X_subset)
        return predictions.mean(axis=1)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.estimators)), dtype=int)
        for i, (tree, feature_indices) in enumerate(self.estimators):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions])



class ForestRegressor(RegressorMixin, BaseForest, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class RandomForestRegressor(ForestRegressor):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.estimators_ = []
        self.feature_importances_ = None


    def fit(self, X, y):
        for _ in range(self.n_estimators):
            estimator = DecisionTreeRegressor.DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            estimator.fit(X[indices], y[indices])
            self.estimators_.append(estimator)

        self.feature_importances_ = self._compute_feature_importances(X)

    def _compute_feature_importances(self, X):
        """
        Compute feature importances for the random forest.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.

        Returns:
        array-like of shape (n_features,)
            The computed feature importances.
        """
        importances = np.zeros(X.shape[1])

        for tree in self.estimators_:
            importances += np.asarray(tree.feature_importances_, dtype=np.float64)

        importances /= len(self.estimators_)
        return importances

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)