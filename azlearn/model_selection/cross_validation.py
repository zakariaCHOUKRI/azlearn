def k_fold_split(X, y, n_splits=5, shuffle=True, random_state=None):
  """
  Splits data into k folds for cross-validation.

  Parameters:
      X (numpy.ndarray): The data to split.
      y (numpy.ndarray): The target variable.
      n_splits (int, default=5): Number of folds for cross-validation.
      shuffle (bool, default=True): Whether to shuffle data before splitting.
      random_state (int, default=None): Seed for random shuffling.

  Returns:
      list: List of tuples containing training and testing indices for each fold.
  """
  if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
    raise TypeError("Expected numpy.ndarray for X and y")
  if len(X) != len(y):
    raise ValueError("X and y must have the same number of samples")
  if n_splits <= 0:
    raise ValueError("n_splits must be positive")

  if shuffle:
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

  fold_size = int(len(X) / n_splits)
  folds = []
  for i in range(n_splits):
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(X)
    test_indices = slice(start_idx, end_idx)
    train_indices = np.concatenate((slice(0, start_idx), slice(end_idx, None)))
    folds.append((train_indices, test_indices))
  return folds

def cross_val_score(estimator, X, y, cv=5, scoring=None):
  """
  Evaluate a score by cross-validation.

  The default scoring metric is 'accuracy' for classification and 'r2' for regression.

  Parameters:
      estimator (object): The estimator object to use for cross-validation.
      X (numpy.ndarray): The data to use for cross-validation.
      y (numpy.ndarray): The target variable for cross-validation.
      cv (int, default=5): Number of folds for cross-validation.
      scoring (callable or str, default=None): A callable scoring function or a pre-defined metric string.

  Returns:
      numpy.ndarray: Array of scores from each fold.
  """
  if not callable(scoring) and scoring is not None:
    if hasattr(estimator, "predict_proba"):
      scoring = "accuracy"
    else:
      scoring = "r2"

  folds = k_fold_split(X, y, cv, shuffle=True)
  scores = np.zeros(len(folds))
  for i, (train_idx, test_idx) in enumerate(folds):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    estimator.fit(X_train, y_train)
    scores[i] = metrics.score(estimator, X_test, y_test, scoring=scoring)
  return scores
