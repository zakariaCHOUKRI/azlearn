import sys
sys.path.insert(1, '..')

from azlearn.base import BaseRegressor, RegressorMixin
import numpy as np

class DecisionTreeRegressor(BaseRegressor, RegressorMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _mse(self, y):
        """
        Calculate the mean squared error of a target variable.

        Parameters:
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        float
            Mean squared error of the target variable.
        """
        return np.mean((y - np.mean(y)) ** 2)

    def _mean_absolute_deviation(self, y):
        """
        Calculate the mean absolute deviation of a target variable.

        Parameters:
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        float
            Mean absolute deviation of the target variable.
        """
        return np.mean(np.abs(y - np.mean(y)))

    def _variance_reduction(self, y, y_left, y_right):
        """
        Calculate the reduction in variance of a split.

        Parameters:
        y : array-like of shape (n_samples,)
            The target values of the parent node.
        y_left : array-like of shape (n_samples,)
            The target values of the left child node.
        y_right : array-like of shape (n_samples,)
            The target values of the right child node.

        Returns:
        float
            Reduction in variance.
        """
        n = len(y)
        variance_parent = np.var(y)
        variance_left = np.var(y_left) * len(y_left) / n
        variance_right = np.var(y_right) * len(y_right) / n
        variance_reduction = variance_parent - (variance_left + variance_right)
        return variance_reduction

    def _find_best_split(self, X, y):
        """
        Find the best split for the dataset.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        tuple
            Best feature index and threshold for splitting, along with the maximum variance reduction.
        """
        best_feature_index = None
        best_threshold = None
        max_variance_reduction = -np.inf

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                y_left = y[X[:, feature_index] <= threshold]
                y_right = y[X[:, feature_index] > threshold]
                variance_reduction = self._variance_reduction(y, y_left, y_right)
                if variance_reduction > max_variance_reduction:
                    max_variance_reduction = variance_reduction
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold, max_variance_reduction

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        depth : int, optional (default=0)
            Current depth of the tree.

        Returns:
        dict
            The constructed decision tree.
        """
        if depth == self.max_depth or len(y) < self.min_samples_split:
            return {'value': np.mean(y)}

        best_feature_index, best_threshold, max_variance_reduction = self._find_best_split(X, y)

        if max_variance_reduction == 0 or len(y) < 2 * self.min_samples_leaf:
            return {'value': np.mean(y)}

        left_mask = X[:, best_feature_index] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature_index': best_feature_index,
                'threshold': best_threshold,
                'left': left_subtree,
                'right': right_subtree}

    def fit(self, X, y):
        """
        Build a decision tree regressor from the training set (X, y).

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        self : object
            Returns self.
        """
        self.tree = self._build_tree(X, y)
        return self

    def _predict_sample(self, x, tree):
        """
        Predict the value of a single sample using the decision tree.

        Parameters:
        x : array-like of shape (n_features,)
            The input sample.
        tree : dict
            The decision tree node.

        Returns:
        float
            The predicted value.
        """
        if 'value' in tree:
            return tree['value']
        else:
            feature_index = tree['feature_index']
            threshold = tree['threshold']
            if x[feature_index] <= threshold:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])

    def predict(self, X):
        """
        Predict target values for samples in X.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.

        Returns:
        array-like of shape (n_samples,)
            The predicted target values.
        """
        if self.tree is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        return np.array([self._predict_sample(x, self.tree) for x in X])
