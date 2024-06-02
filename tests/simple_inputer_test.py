import numpy as np
import sys
sys.path.insert(1, '../azlearn')
from azlearn.preprocessing.imputation import SimpleImputer

# Test for SimpleImputer
def simple_imputer_test():
    # Test data with missing values
    X = np.array([[1, np.nan, 3],
                  [4, 5, np.nan],
                  [7, 8, 9]])

    # Instantiate and fit the SimpleImputer with mean strategy
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X)

    # Check if the imputer has computed the correct mean
    assert np.allclose(imputer.statistics_, [4, 6.5, 6])

    # Transform the dataset using the fitted imputer
    X_transformed = imputer.transform(X)

    # Expected result after mean imputation
    X_expected = np.array([[1, 6.5, 3],
                            [4, 5, 6],
                            [7, 8, 9]])

    # Check if the imputed dataset matches the expected result
    assert np.array_equal(X_transformed, X_expected)

    print(X_transformed)

# Run the test
simple_imputer_test()