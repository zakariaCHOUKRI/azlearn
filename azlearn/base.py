class BaseEstimator:
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    def score(self, X, y):
        raise NotImplementedError
    
    def get_params(self):
        raise NotImplementedError
    
    def set_params(self, **params):
        raise NotImplementedError

class BaseClassifier(BaseEstimator):
    def predict_proba(self, X):
        raise NotImplementedError

class BaseRegressor(BaseEstimator):
    pass

class BaseCluster(BaseEstimator):
    def fit_predict(self, X):
        raise NotImplementedError

class TransformerMixin:
    def fit_transform(self, X, y=None):
        raise NotImplementedError
    
    def transform(self, X):
        raise NotImplementedError
    
    def inverse_transform(self, X):
        raise NotImplementedError

class BaseCrossValidator:
    def split(self, X, y=None, groups=None):
        raise NotImplementedError
    
    def get_n_splits(self, X, y=None, groups=None):
        raise NotImplementedError

class BaseGridSearchCV:
    def __init__(self, estimator, param_grid, cv=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
    
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
