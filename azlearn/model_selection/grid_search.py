from copy import deepcopy

sys.path.insert(1, '../azlearn')
from azlearn.base import BaseEnsemble,MetaEstimatorMixin

class GridSearchCV(BaseEstimator, MetaEstimatorMixin):

  def __init__(self, estimator, param_grid, cv=5, scoring=None, refit=True):
    self.estimator = estimator
    self.param_grid = param_grid
    self.cv = cv
    self.scoring = scoring
    self.refit = refit
    self.best_estimator_ = None
    self.best_score_ = None
    self.cv_results_ = None

  def _fit(self, X, y):
    """
    Fits the grid of hyperparameters using cross-validation.

    Args:
        X (numpy.ndarray): The data to use for fitting.
        y (numpy.ndarray): The target variable.
    """
    self.cv_results_ = {'param_grid': self.param_grid, 'cv_results': []}
    for params in self.param_grid:
      estimator_clone = deepcopy(self.estimator)  # Deepcopy to avoid modifying original
      estimator_clone.set_params(**params)
      cv_results = cross_val_score(estimator_clone, X, y, cv=self.cv, scoring=self.scoring)
      self.cv_results_['cv_results'].append({
          'params': params,
          'mean_test_score': cv_results.mean(),
          'std_test_score': cv_results.std()
      })
    best_idx = np.argmax([result['mean_test_score'] for result in self.cv_results_['cv_results']])
    self.best_estimator_ = deepcopy(self.estimator)
    self.best_estimator_.set_params(**self.param_grid[best_idx])
    self.best_score_ = self.cv_results_['cv_results'][best_idx]['mean_test_score']
    if self.refit:
      self.best_estimator_.fit(X, y)

  def fit(self, X, y):
    """
    Fits the grid search using cross-validation.

    Args:
        X (numpy.ndarray): The data to use for fitting.
        y (numpy.ndarray): The target variable.
    """
    self._fit(X, y)
    return self

  def predict(self):
    """
    Predicts using the best model found during grid search.

    Raises:
        AttributeError: If `refit` was set to `False` during initialization.
    """
    if not self.refit:
      raise AttributeError("GridSearchCV was not fitted with refit=True. Call fit first.")
    return self.best_estimator_.predict(X)

  def score(self, X, y):
    """
    Evaluates the score of the best model on the given data.

    Raises:
        AttributeError: If `refit` was set to `False` during initialization.
    """
    if not self.refit:
      raise AttributeError("GridSearchCV was not fitted with refit=True. Call fit first.")
    return metrics.score(self.best_estimator_, X, y, scoring=self.scoring)

  def get_params(self, deep=True):
    """
    Returns parameters for grid search and nested estimator.
    """
    out = super().get_params(deep=deep)
    out.update({'estimator': self.estimator.get_params(deep=deep),
                'param_grid': self.param_grid,
                'cv': self.cv,
                'scoring': self.scoring,
                'refit': self.refit})
    return out

  def set_params(self, **params):
    """
    Sets parameters for grid search and nested estimator.
    """
    super().set_params(**params)
    self.estimator = clone(params.get('estimator'))
    self.param_grid = params['param_grid']
    self.cv = params['cv']
    self.scoring = params['scoring']
    self.refit = params['refit']
    return self

  def cv_results_(self):
    """
    Returns the cross-validation results.

    Raises:
        AttributeError: If `fit` has not been called yet.
    """
    if self.cv_results_ is None :
        raise AttributeError("GridSearchCV must be fitted before accessing cv_results_")
    return self.cv_results_
