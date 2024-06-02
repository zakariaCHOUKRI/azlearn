import numpy as np
import inspect
import sys

sys.path.insert(1, '../azlearn')
from abc import ABCMeta, abstractmethod


class BaseEstimator:

    @classmethod
    def get_parameters_names(cls):
        constructor_signature = inspect.signature(cls.__init__)
        # Extract parameter names
        parameter_names = list(constructor_signature.parameters.keys())
        return sorted(parameter_names)

    def get_params(self):
        """
        returns dict {name_of_paramater: value_of_parameter}
        """
        names_dict = {}
        for name in self.get_parameters_names():
            names_dict[name] = getattr(self, name, None)
        return names_dict

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class Predictor(BaseEstimator):

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class Transformer(BaseEstimator):

    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        """Reverse the transformation."""
        raise NotImplementedError("inverse_transform method not implemented")


class BaseClassifier:
    def fit(self, X, y):
        raise NotImplementedError("fit method not implemented")

    def predict(self, X):
        raise NotImplementedError("predict method not implemented")

    @classmethod
    def get_parameters_names(cls):
        constructor_signature = inspect.signature(cls.__init__)
        # Extract parameter names
        parameter_names = list(constructor_signature.parameters.keys())
        return sorted(parameter_names)

    def get_params(self):
        """
        returns dict {name_of_paramater: value_of_parameter}
        """
        names_dict = {}
        for name in self.get_parameters_names():
            names_dict[name] = getattr(self, name, None)
        return names_dict

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self



class BaseRegressor(Predictor):
    pass


class ClassifierMixin:
    def score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


class RegressorMixin:
    def score(self, y_true, y_pred):
        return r2_score(y_true, y_pred)


class MetaEstimatorMixin:
    pass


class TransformerMixin:
    pass


class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, estimators, **kwargs):
        self.estimators = estimators
        self.set_params(**kwargs)

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        params = {}
        for key in self.__dict__:
            if deep and hasattr(self.__dict__[key], "get_params"):
                nested_params = self.__dict__[key].get_params(deep=True)
                params.update({key + "__" + k: v for k, v in nested_params.items()})
            else:
                params[key] = self.__dict__[key]
        return params


def clone(estimator):
    params = estimator.get_params()
    return estimator.__class__(**params)









