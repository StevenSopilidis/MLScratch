import numpy as np
import math
from Functions.loss_functions import Sigmoid

class LogisticRegression():
    """
    Class that implements Logistic regression
    using batch gradient descent
    
    Parameters:
    -------
    n_iterations: number of iterations algorithm will take
    learning_rate: learning rate of the algorithm
    """

    def __init__(self, n_iterations: int = 100, learning_rate: int = 0.1):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X: np.array):
        """
        Method for initializing the weights of the model

        Parameters
        -------
        X: dataset
        """
        n_features = np.shape(X)[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        self._initialize_parameters(X)
        for _ in range(self.n_iterations):
            y_pred = self.sigmoid(X.dot(self.weights))
            self.weights -= self.learning_rate * -(y - y_pred).dot(X)
            self.bias -= self.learning_rate * (-(y - y_pred) * self.bias)

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.weights) + self.bias)).astype(int)
        return y_pred