import numpy as np
import sys
sys.path.insert(1, '../azlearn')

from preprocessing.encoding import OneHotEncoder,LabelEncoder
from pipeline import Pipeline

one_hot_encoder = OneHotEncoder()
label_encoder = LabelEncoder()

# Define the pipeline
pipe = Pipeline(steps=[('one_hot', one_hot_encoder), ])

# Test the pipeline
data = np.array(['cat', 'dog', 'cat', 'bird'])
transformed_data = pipe.fit_transform(data)
print("Transformed data:")
print(transformed_data)