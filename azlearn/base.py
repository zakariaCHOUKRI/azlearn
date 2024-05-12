import inspect
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
        names_dict =  {}
        for name in self.get_parameters_names():
            out[name] = getattr(self, name,None)

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


class ClassifierMixin :
    def score(self):
        raise NotImplementedError

class RegressorMixin:

    def score(self):
        raise NotImplementedError



class MetaEstimatorMixin :
        pass
