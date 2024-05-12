import sys
sys.path.insert(1, '..')

from azlearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from abc import ABC, abstractmethod

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)



class BaseMultilayerPerceptron(BaseEstimator):
    @abstractmethod
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    @abstractmethod
    def _initialize_weights(self, n_features, n_classes):
        pass

    @abstractmethod
    def _forward_pass(self, X):
        pass

    @abstractmethod
    def _backward_pass(self, X, y):
        pass

    @abstractmethod
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self._initialize_weights(n_features, n_classes)
        for _ in range(self.max_iter):
            self._forward_pass(X)
            self._backward_pass(X, y)
        return self

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass


class MLPClassifier(BaseMultilayerPerceptron, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001, max_iter=200):
        super().__init__(hidden_layer_sizes, activation, learning_rate, max_iter)
        self.weights = []

    def _initialize_weights(self, n_features, n_classes):
        # Initialize weights randomly
        for i in range(len(self.hidden_layer_sizes)):
            if i == 0:
                prev_size = n_features
            else:
                prev_size = self.hidden_layer_sizes[i - 1]
            self.weights.append(np.random.randn(prev_size, self.hidden_layer_sizes[i]))
        self.weights.append(np.random.randn(self.hidden_layer_sizes[-1], n_classes))

    def _forward_pass(self, X):
        self.layer_outputs = []
        out = X
        for i in range(len(self.weights)):
            out = self._activation_function(np.dot(out, self.weights[i]))
            self.layer_outputs.append(out)
        self.output = out

    def _backward_pass(self, X, y):
        n_samples = X.shape[0]
        d_weights = [np.zeros_like(w) for w in self.weights]
        error = self.output - np.eye(len(self.classes_))[y]
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                delta = error
            else:
                delta = np.dot(delta, self.weights[i + 1].T)
            delta *= self._activation_derivative(self.layer_outputs[i])
            d_weights[i] = np.dot(self.layer_outputs[i - 1].T, delta) / n_samples
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * d_weights[i]

    def _activation_function(self, X):
        if self.activation == 'relu':
            return np.maximum(0, X)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        else:
            raise ValueError("Unknown activation function.")

    def _activation_derivative(self, X):
        if self.activation == 'relu':
            return (X > 0).astype(float)
        elif self.activation == 'sigmoid':
            return X * (1 - X)
        else:
            raise ValueError("Unknown activation function.")

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        self._forward_pass(X)
        return softmax(self.output)