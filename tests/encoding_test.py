import sys
sys.path.insert(1, '../azlearn')

from preprocessing.encoding import OneHotEncoder, LabelEncoder
import numpy as np


data = np.array(['cat', 'dog', 'cat', 'bird'])

# One-hot encoding
one_hot_encoder = OneHotEncoder()
one_hot_encoded = one_hot_encoder.fit_transform(data)
print("One-hot encoded data:")
print(one_hot_encoded)

# Label encoding
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(data)
print("\nLabel encoded data:")
print(label_encoded)