import numpy as np
from azlearn.base import BaseClassifier, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


class AdaBoostClassifier(BaseClassifier, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

    def fit(self, X, y):
        """
        Build a boosted classifier from the training set (X, y).

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        self : object
            Returns self.
        """
        n_samples, _ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        sample_weights = np.ones(n_samples) / n_samples

        for iboost in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            estimator.fit(X, y, sample_weight=sample_weights)
            y_pred = estimator.predict(X)

            incorrect = (y_pred != y)
            estimator_error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)

            if estimator_error >= 1.0 - (1.0 / self.n_classes_):
                break

            alpha = self.learning_rate * np.log((1.0 - estimator_error) / estimator_error)
            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= np.sum(sample_weights)

            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(estimator_error)

        return self

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
        pred = sum(
            estimator.predict(X) * weight for estimator, weight in zip(self.estimators_, self.estimator_weights_))
        pred /= np.sum(self.estimator_weights_)
        pred = np.sign(pred)
        return pred

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
        proba = sum(
            estimator.predict_proba(X) * weight for estimator, weight in zip(self.estimators_, self.estimator_weights_))
        proba /= np.sum(self.estimator_weights_)
        return proba

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
