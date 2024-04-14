import numpy as np
from utils.data_operations import accuracy_score

class SquareLoss():
    """
    Class that represents square loss function
    y: actual values
    y_pred: predicted values
    """
    def __init__(self) -> None: pass

    def loss(self, y: np.array, y_pred: np.array) -> float:
        return 0.5 * np.power((y - y_pred), 2)
    
    def gradient(self, y: np.array, y_pred: np.array) -> float:
        return -(y - y_pred)
    
class CrossEntropy():
    """
    Class that represents CrossEntropy loss function
    """
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)