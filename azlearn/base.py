import inspect
import sys

sys.path.insert(1, '../azlearn')
from metrics.metrics import accuracy_score, r2_score
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
            out[name] = getattr(self, name, None)

        return names_dict

    def set_params(self, **params):
        raise NotImplementedError


class Predictor(BaseEstimator):

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def fit_predict(self, X, y):
        raise NotImplementedError


class Transformer(BaseEstimator):

    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y):
        raise NotImplementedError

    def inverse_transform(self, X):
        """Reverse the transformation."""
        raise NotImplementedError("inverse_transform method not implemented")


class BaseClassifier:
    def fit(self, X, y):
        """
        Fit classifier to the training data.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        raise NotImplementedError("fit method not implemented")

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Samples.

        Returns:
        array-like of shape (n_samples,)
            Predicted class labels.
        """
        raise NotImplementedError("predict method not implemented")



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

class TransformerMixin :
    pass


class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Base class for ensemble methods.

    This class inherits from MetaEstimatorMixin and BaseEstimator,
    allowing for scikit-learn compatibility.
    """

    @abstractmethod
    def __init__(self, estimators, **kwargs):
        """
        Initialize the ensemble method.

        Parameters:
        -----------
        estimators : list of estimators
            List of base estimators for the ensemble.

        **kwargs:
            Additional keyword arguments specific to the ensemble method.
        """
        self.estimators = estimators
        self.set_params(**kwargs)

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the ensemble method to the training data.

        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict using the ensemble method.

        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            The predicted target values.
        """
        pass

    def set_params(self, **params):
        """
        Set the parameters of the ensemble method.

        Parameters:
        -----------
        **params:
            Additional keyword arguments.

        Returns:
        --------
        self : object
            Returns self.
        """
        if not params:
            return self
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of the ensemble method.

        Parameters:
        -----------
        deep : bool, default=True
            If True, return the parameters as nested dictionaries.

        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        params = {}
        for key in self.__dict__:
            if deep and hasattr(self.__dict__[key], "get_params"):
                nested_params = self.__dict__[key].get_params(deep=True)
                params.update({key + "__" + k: v for k, v in nested_params.items()})
            else:
                params[key] = self.__dict__[key]
        return params