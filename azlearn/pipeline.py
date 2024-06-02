import sys
sys.path.insert(1, '../azlearn')

from preprocessing.encoding import _BaseEncoder
class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit_transform(self, X, y=None):
        for step in self.steps:
            if isinstance(step[1],_BaseEncoder):
                X = step[1].fit_transform(X)
            else :
                X   = step[1].fit_transform(X, y)
        return X

