sys.path.insert(1, '../azlearn')
from azlearn.base import Transformer


class StandardScaler(Transformer):
  """
  StandardScaler normalizes features using the formula:
  X_scaled = (X - mean) / std
  where X is the input data, mean is the mean of each feature,
  and std is the standard deviation of each feature.

  Parameters:
      with_mean (bool, default=True): Center features by removing the mean
          across all samples.
      with_std (bool, default=True): Scale features by dividing by the standard
          deviation across all samples.
  """

  def __init__(self, with_mean=True, with_std=True):
    self.with_mean = with_mean
    self.with_std = with_std
    self.means_ = None
    self.stds_ = None

  def fit(self, X):
    """
    Computes the mean and standard deviation of each feature.

    Args:
        X (numpy.ndarray): The input data.

    Returns:
        StandardScaler: The fitted transformer object.
    """
    if not isinstance(X, np.ndarray):
      raise TypeError("Expected numpy.ndarray, got {}".format(type(X)))
    self.means_ = np.mean(X, axis=0)
    self.stds_ = np.std(X, axis=0)
    # Handle division by zero for features with zero std
    self.stds_[self.stds_ == 0] = 1e-10
    return self

  def transform(self, X):
    """
    Normalizes the features based on the precomputed mean and standard deviation.

    Args:
        X (numpy.ndarray): The input data.

    Returns:
        numpy.ndarray: The transformed data.
    """
    if not isinstance(X, np.ndarray):
      raise TypeError("Expected numpy.ndarray, got {}".format(type(X)))
    if self.means_ is None or self.stds_ is None:
      raise ValueError("This StandardScaler instance is not fitted yet. "
                       "Call 'fit' with appropriate arguments before using this method.")
    X_scaled = X.copy()
    if self.with_mean:
      X_scaled -= self.means_
    if self.with_std:
      X_scaled /= self.stds_
    return X_scaled

class Normalizer(Transformer):
      """
      Normalizer normalizes each sample individually to unit norm.

      Parameters:
          norm (str, optional): The norm to use to normalize each non-zero sample.
              Valid options are 'l1', 'l2' (default), or 'max'.
      """

      def __init__(self, norm='l2'):
          self.norm = norm

      def fit(self, X):
          """
          This method does nothing for normalization as it doesn't require fitting parameters.

          Args:
              X (numpy.ndarray): The input data (ignored).

          Returns:
              Normalizer: The fitted transformer object (itself).
          """
          return self

      def transform(self, X):
          """
          Normalizes each sample in the input data.

          Args:
              X (numpy.ndarray): The input data.

          Returns:
              numpy.ndarray: The transformed data.
          """
          if not isinstance(X, np.ndarray):
              raise TypeError("Expected numpy.ndarray, got {}".format(type(X)))
          if self.norm not in ['l1', 'l2', 'max']:
              raise ValueError("Unsupported norm '{}'. Valid options are 'l1', 'l2', or 'max'.".format(self.norm))

          norms = np.linalg.norm(X, axis=1, ord=self.norm)
          # Handle division by zero for samples with zero norm
          norms[norms == 0] = 1e-10
          X_normalized = X / np.expand_dims(norms, axis=1)
          return X_normalized