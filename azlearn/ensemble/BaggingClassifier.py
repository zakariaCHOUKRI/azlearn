import sys
sys.path.insert(1, '../azlearn')
from azlearn.base import BaseEnsemble,ClassifierMixin,clone
import numpy as np
from azlearn.metrics.metrics import *

class BaggingClassifier(BaseEnsemble, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []
        super().__init__(estimators=[base_estimator for _ in range(n_estimators)])

    def fit(self, X, y):
        self.estimators_ = []
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            estimator = self.base_estimator
            X_resampled, y_resampled = resample(X, y, n_samples=int(self.max_samples * len(X)),
                                                random_state=rng.randint(1e6))
            estimator.fit(X_resampled, y_resampled)
            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        predictions = np.asarray([estimator.predict(X) for estimator in self.estimators_])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_vote

    def predict_proba(self, X):
        probas = np.asarray([estimator.predict_proba(X) for estimator in self.estimators_])
        avg_probas = np.mean(probas, axis=0)
        return avg_probas

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


def resample(X, y, n_samples=None, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = n_samples or X.shape[0]
    indices = np.random.choice(np.arange(X.shape[0]), size=n_samples, replace=True)

    X_resampled = X[indices]
    y_resampled = y[indices]

    return X_resampled, y_resampled