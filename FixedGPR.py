import numpy as np

class RBF:

    def __init__(self, variance, length_scale):
        self.variance = variance
        self.length_scale = length_scale

    def __call__(self, x, y, *args, **kwargs):
        return self.variance * np.exp(-np.linalg.norm(x - y)**2/(2 * self.length_scale))

class FixedGPR:

    def __init__(self, kernel):
        pass

    def predict(self, X):
        pass

    def fit(self, X, y):
        pass

