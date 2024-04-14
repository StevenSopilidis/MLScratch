from functions.activation_functions import Sigmoid, Softmax
from functions.loss_functions import CrossEntropy
import numpy as np


class MLP:
    """
    Class that implements multilayer perceptron using one hidden layer

    Parameters
    -------
    n_hidden: number of perceptrons in hidden layer
    n_iterations: number of iterations training will take
    learning_rate: learning rate of the algorithm
    """
    def __init__(self, n_hidden: int, n_iterations: int = 3000, learning_rate: float = 0.01) -> None:
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()
    
    def _init_weights(self, X: np.array, y: np.array) -> None:
        """
        Funtion for initializing the weights between [-1/n, 1/n]
        where n == number of features

        Parameters:
        -------
        X: training_data
        y: labels
        """
        
        n_samples, n_features = X.shape
        _, n_outputs = y.shape

        # init hidden layer
        limit = 1/n_features
        self.W = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))

        # init output layer
        limit = 1/self.n_hidden
        self.V = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs))
        self.v0 = np.zeros((1, n_outputs))


    def fit(self, X: np.array, y: np.array) -> None:
        self._init_weights(X, y)

        for _ in range(self.n_iterations):
            
            # ---- ForwardsPass ----
            hidden_input = X.dot(self.W) + self.w0
            hidden_output = self.hidden_activation(hidden_input)

            output_layer_input = hidden_output.dot(self.V) + self.v0
            y_pred = self.output_activation(output_layer_input)

            # ---- BackwardsPass ----
            grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            grad_v = hidden_output.T.dot(grad_wrt_out_l_input)
            grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)

            grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_activation.gradient(hidden_input)
            grad_w = X.T.dot(grad_wrt_hidden_l_input)
            grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

            self.V  -= self.learning_rate * grad_v
            self.v0 -= self.learning_rate * grad_v0
            self.W  -= self.learning_rate * grad_w
            self.w0 -= self.learning_rate * grad_w0

    def predict(self, X):
        hidden_input = X.dot(self.W) + self.w0
        hidden_output = self.hidden_activation(hidden_input)
        output_layer_input = hidden_output.dot(self.V) + self.v0
        y_pred = self.output_activation(output_layer_input)
        return y_pred
