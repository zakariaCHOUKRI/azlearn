from random import randint
import numpy as np


def generate_synthetic_data(n_samples=1000, random_state=None):
    rng = np.random.RandomState(random_state)
    X = rng.randint(0, 3, size=(n_samples, 3))  # features "spam", "ham", "other" represented as 0, 1, 2
    y = rng.randint(0, 2, size=n_samples)  # binary classification
    return X, y



def generate_synthetic_data_regression(n_samples=1000, random_state=None):
    rng = np.random.RandomState(random_state)
    X = rng.randint(0, 3, size=(n_samples, 3))  # features "spam", "ham", "other" represented as 0, 1, 2
    y = rng.rand(n_samples) * 10  # continuous target variable
    return X, y
