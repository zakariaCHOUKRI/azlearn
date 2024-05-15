import sys
sys.path.insert(1, '..')

from azlearn.base import BaseClassifier, ClassifierMixin
import numpy as np

class DecisionTreeClassifier(BaseClassifier, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _entropy(self, y):
        """
        Calculate the entropy of a target variable.

        Parameters:
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        float
            Entropy of the target variable.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _information_gain(self, X, y, feature_index, threshold):
        """
        Calculate the information gain of splitting on a feature and threshold.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        feature_index : int
            Index of the feature to split on.
        threshold : float
            Threshold value for splitting the feature.

        Returns:
        float
            Information gain of the split.
        """
        # Split data based on the threshold
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        # Calculate parent entropy
        parent_entropy = self._entropy(y)

        # Calculate children entropy
        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])

        # Calculate information gain
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        total_samples = len(y)
        information_gain = parent_entropy - ((n_left / total_samples) * left_entropy + (n_right / total_samples) * right_entropy)

        return information_gain

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
            Best feature index and threshold for splitting, along with the maximum information gain.
        """
        best_feature_index = None
        best_threshold = None
        max_information_gain = -np.inf

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                information_gain = self._information_gain(X, y, feature_index, threshold)
                if information_gain > max_information_gain:
                    max_information_gain = information_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold, max_information_gain

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
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return {'class': np.argmax(np.bincount(y))}

        best_feature_index, best_threshold, max_information_gain = self._find_best_split(X, y)

        if max_information_gain == 0:
            return {'class': np.argmax(np.bincount(y))}

        left_mask = X[:, best_feature_index] <= best_threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return {'class': np.argmax(np.bincount(y))}

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature_index': best_feature_index,
                'threshold': best_threshold,
                'left': left_subtree,
                'right': right_subtree}

    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        self : object
            Returns self.
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.tree = self._build_tree(X, y)
        return self

    def _predict_sample(self, x, tree):
        """
        Predict the class label of a single sample using the decision tree.

        Parameters:
        x : array-like of shape (n_features,)
            The input sample.
        tree : dict
            The decision tree node.

        Returns:
        int
            The predicted class label.
        """
        if 'class' in tree:
            return tree['class']
        else:
            feature_index = tree['feature_index']
            threshold = tree['threshold']
            if x[feature_index] <= threshold:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.

        Returns:
        array-like of shape (n_samples,)
            The predicted class labels.
        """
        if self.tree is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        return np.array([self._predict_sample(x, self.tree) for x in X])

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.

        Returns:
        array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        if self.tree is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        def _predict_proba_sample(x, tree):
            if 'class' in tree:
                class_probabilities = np.zeros(self.n_classes_)
                class_probabilities[tree['class']] = 1
                return class_probabilities
            else:
                feature_index = tree['feature_index']
                threshold = tree['threshold']
                if x[feature_index] <= threshold:
                    return _predict_proba_sample(x, tree['left'])
                else:
                    return _predict_proba_sample(x, tree['right'])

        return np.array([_predict_proba_sample(x, self.tree) for x in X])
