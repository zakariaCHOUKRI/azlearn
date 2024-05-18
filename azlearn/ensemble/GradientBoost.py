import numpy as np
from azlearn.base import BaseClassifier, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone

class GradientBoostingClassifier(BaseClassifier, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeRegressor(max_depth=max_depth)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []

    def _compute_initial_prediction(self, y):
        # Compute initial prediction as the log odds
        pos_ratio = np.mean(y)
        neg_ratio = 1 - pos_ratio
        return np.log(pos_ratio / neg_ratio)

    def _compute_negative_gradient(self, y, y_pred):
        # Compute negative gradient for logistic loss
        return y - 1 / (1 + np.exp(-y_pred))

    def fit(self, X, y):
        """
        Build a gradient boosting classifier from the training set (X, y).

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        self : object
            Returns self.
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        y_transformed = np.where(y == self.classes_[0], -1, 1).astype(np.float64)

        # Initial prediction
        initial_prediction = self._compute_initial_prediction(y_transformed)
        self.init_pred_ = initial_prediction

        self.estimators_ = []
        residuals = y_transformed.copy()

        for i in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            estimator.fit(X, residuals)
            y_pred = estimator.predict(X)

            residuals -= self.learning_rate * y_pred
            self.estimators_.append(estimator)

        return self

    def _predict_raw(self, X):
        raw_pred = np.full(X.shape[0], self.init_pred_, dtype=np.float64)
        for estimator in self.estimators_:
            raw_pred += self.learning_rate * estimator.predict(X)
        return raw_pred

    def predict(self, X):
        """
        Predict classes for X.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        y_pred : array-like of shape (n_samples,)
            The predicted classes.
        """
        raw_pred = self._predict_raw(X)
        return np.where(raw_pred > 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        proba : array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        raw_pred = self._predict_raw(X)
        proba_pos = 1 / (1 + np.exp(-raw_pred))
        return np.vstack([1 - proba_pos, proba_pos]).T

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns:
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
