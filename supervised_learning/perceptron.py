import numpy as np
import math
from functions.loss_functions import SquareLoss
from functions.activation_functions import Sigmoid


class Perceptron:
    """
    Parameters
    -------
    n_iterations: number of iterations that algorithm will run for
    activation_function: activation function that will be used for TLU
    loss: loss function that will be used to tune weights
    learning_rate: learning rate of algorithm
    """
    def __init__(self, n_iterations: int=20000, activation_function=Sigmoid, loss=SquareLoss, learning_rate: float=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.activation_function = activation_function()
        self.loss = loss()

    def fit(self, X, y):
        _, n_features = np.shape(X)
        _, n_outputs = np.shape(y)
        
        limit = 1 / math.sqrt(n_features)
        # init weights using uniform distribution
        self.W = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.bias = np.zeros((1, n_outputs))

        for i in range(self.n_iterations):
            dot = X.dot(self.W) + self.bias
            y_pred = self.activation_function(dot)

            # calculate error_gradient of prediction
            error_gradient = self.loss.gradient(y, y_pred) * self.activation_function.gradient(dot)
            # update weights
            w_gradient = X.T.dot(error_gradient)
            bias_gradient = np.sum(error_gradient, axis=0, keepdims=True)

            self.W -= self.learning_rate * w_gradient
            self.bias -= self.learning_rate * bias_gradient

    def predict(self, X):
        return self.activation_function(X.dot(self.W) + self.bias)