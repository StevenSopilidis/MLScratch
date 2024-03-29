import numpy as np

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