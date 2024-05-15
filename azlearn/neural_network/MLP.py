import sys
sys.path.insert(1, '..')

from azlearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np
from abc import ABC, abstractmethod

class BaseMultilayerPerceptron(BaseEstimator):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _initialize_weights(self, input_size, output_size):
        # Initialize weights with small random values
        self.weights = []
        layer_sizes = [input_size] + list(self.hidden_layer_sizes) + [output_size]
        for i in range(1, len(layer_sizes)):
            weight_matrix = np.random.randn(layer_sizes[i-1], layer_sizes[i])
            self.weights.append(weight_matrix)

    def _activation_function(self, z):
        # Activation function (ReLU or sigmoid)
        if self.activation == 'relu':
            return np.maximum(z, 0)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))

    def _forward_pass(self, X):
        # Perform forward pass through the network
        activations = [X]
        for weight_matrix in self.weights:
            z = np.dot(activations[-1], weight_matrix)
            activation = self._activation_function(z)
            activations.append(activation)
        return activations

    def _backward_pass(self, X, y, activations):
        # Perform backward pass through the network (backpropagation)
        gradients = []
        output_layer_delta = activations[-1] - y
        gradients.append(np.dot(activations[-2].T, output_layer_delta))
        for i in range(len(self.weights)-1, 0, -1):
            delta = np.dot(output_layer_delta, self.weights[i].T) * (activations[i] > 0 if self.activation == 'relu' else activations[i] * (1 - activations[i]))
            gradients.insert(0, np.dot(activations[i-1].T, delta))
            output_layer_delta = delta
        return gradients

    def fit(self, X, y):
        # Train the model
        n_samples, input_size = X.shape
        _, output_size = y.shape
        self._initialize_weights(input_size, output_size)
        for _ in range(self.max_iter):
            activations = self._forward_pass(X)
            gradients = self._backward_pass(X, y, activations)
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients[i]

    def predict(self, X):
        # Predict the labels for input X
        activations = self._forward_pass(X)
        return np.argmax(activations[-1], axis=1)

class MLPClassifier(ClassifierMixin, BaseMultilayerPerceptron):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001, max_iter=200):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation, learning_rate=learning_rate, max_iter=max_iter)

    def _one_hot_encode(self, y, num_classes):
        # One-hot encode the target labels
        n_samples = y.shape[0]
        encoded = np.zeros((n_samples, num_classes))
        for i in range(n_samples):
            encoded[i, y[i]] = 1
        return encoded

    def fit(self, X, y):
        # Train the classifier
        num_classes = len(np.unique(y))
        encoded_y = self._one_hot_encode(y, num_classes)
        super().fit(X, encoded_y)

    def _softmax(self, z):
        # Softmax function for output layer
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict_proba(self, X):
        # Predict class probabilities for input X
        activations = self._forward_pass(X)
        return self._softmax(activations[-1])

    def predict(self, X):
        # Predict the class labels for input X
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


class MLPRegressor(RegressorMixin, BaseMultilayerPerceptron):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001, max_iter=200):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation, learning_rate=learning_rate, max_iter=max_iter)

    def fit(self, X, y):
        # Train the regressor
        super().fit(X, y)

    def predict(self, X):
        # Predict the regression targets for input X
        activations = self._forward_pass(X)
        return activations[-1]
