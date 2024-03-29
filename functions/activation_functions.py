import numpy as np

class Sigmoid():
    """
    Class that implemenets sigmoid function
    """
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))