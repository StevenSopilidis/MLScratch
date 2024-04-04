import numpy as np
import math
from enum import Enum

from numpy.core.multiarray import array as array
from utils.data_manipulation import polynomial_features

class Regularization(Enum):
    L1 = "l1"
    L2 = "l2"
    L1_L2 = "l1_l2"
    NOT = "none"

class l1_regularization():
    """
    Class that represents l1_regularization
    
    Paramaters:
    -------
    lamda: regularization param
    """
    def __init__(self, lamda):
        self.lamda = lamda

    def __call__(self, w: np.array) -> float:
        return self.lamda * np.linalg.norm(w)
    
    def grad(self, w: np.array) -> float:
        return self.lamda * np.sign(w)
    
class l2_regularization():
    """
    class that represents l2_regularization

    Parameters:
    -------
    lamda: regularization param
    """

    def __init__(self, lamda):
        self.lamda = lamda
    
    def __call__(self, w: np.array) -> float:
        return self.lamda * 0.5 * w.T.dot(w)
    
    def grad(self, w: np.array) -> float:
        return self.lamda * w

class l1_l2_regularization():
    """
    class that represents l1_l2_regularization

    Parameters:
    -------
    lamda: regularization parameter
    l1_ratio: ratio of l1_regularization
    """
    def __init__(self, lamda, l1_ratio = 0.5) -> None:
        self.lamda = lamda
        self.l1_ratio = l1_ratio

    def __call__(self, w: np.array) -> float:
        l1_term = self.l1_ratio * np.linalg.norm(w)
        l2_term = (1 - self.l1_ratio) * w.T.dot(w)
        return self.lamda * (l1_term + l2_term)
    
    def grad(self, w):
        l1_term = self.l1_ratio * np.sign(w)
        l2_term = (1 - self.l1_ratio) * w
        return self.lamda * (l1_term + l2_term) 
    
class Regression(object):
    """
    Base class for regularization

    Parameters:
    -------
    n_iterations: number of iterations algorithm will take
    learning_rate: learning rate of algorithm
    regularization: regularization function of regression
    """
    def __init__(self, n_iterations: int, learning_rate: float):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def init_weights(self, n_features: int) -> None:
        """
        Function for initialing the weights between [-1/N, 1/n]

        Parameters
        -------
        n_features: number of features of dataset
        """
        limit = 1/math.sqrt(n_features)
        self.weights = np.random.uniform(-limit, limit, (n_features, )) 
    
    def fit(self, X: np.array, y: np.array) -> None:
        X = np.insert(X, 0, 1, axis=1) # insert 1 to dataset to account for bias
        self.init_weights(X.shape[1])

        for _ in range(self.n_iterations):
            y_pred = X.dot(self.weights)
            mse = (y - y_pred)**2 + self.regularization(self.weights)
            grad =  -(y - y_pred).dot(X) + self.regularization.grad(self.weights)
            self.weights -= self.learning_rate * grad
    
    def predict(self, X: np.array) -> np.array:
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.weights)
    
class LinearRegression(Regression):
    """
    Class that represents linear regression algorithm

    Parameters:
    n_iterations: number of iterations algorithm will take
    learning_rate: learning rate of algorithm
    gradient_descent: wether or not we are going to use gradient descent (in case we dont we use least squares)
    regularization: type of regularization to use (l1,l2,none)
    lamda: lamda value of regularization
    l1_ratio: ratio of l1 regularization in case we use l1_l2 ratio
    """
    def __init__(self, n_iterations: int = 100, learning_rate: float = 0.001, gradient_descent: bool = True, regularization: Regularization = Regularization.NOT, lamda: float = 0, l1_ratio: float = 0.5):
        self.gradient_descent = gradient_descent
        if regularization is Regularization.L1:
            self.regularization = l1_regularization(lamda)
        if regularization is Regularization.L2:
            self.regularization = l2_regularization(lamda)
        if regularization is Regularization.L1_L2:
            self.regularization = l1_l2_regularization(lamda, l1_ratio)
        else:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0


        super(LinearRegression, self).__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        if not self.gradient_descent:
            # calculate weights using least squares (using Moore-Penrose pseudoinverse)
            X = np.insert(X, 0, 1, axis=1)
            U,S,V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)


class PolynomialRegression(Regression):
    """
    Class that represents polynomial regression algorithm

    Parameters:
    degree: degree of polynomials
    n_iterations: number of iterations algorithm will take
    learning_rate: learning rate of algorithm
    regularization: type of regularization to use (l1,l2,none)
    lamda: lamda value of regularization
    l1_ratio: ratio of l1 regularization in case we use l1_l2 ratio
    """
    def __init__(self, degree: int = 2, n_iterations: int = 100, learning_rate: float = 0.001, regularization: Regularization = Regularization.NOT, lamda: float = 0, l1_ratio: float = 0.5):
        self.degree = degree
        if regularization is Regularization.L1:
            self.regularization = l1_regularization(lamda)
        if regularization is Regularization.L2:
            self.regularization = l2_regularization(lamda)
        if regularization is Regularization.L1_L2:
            self.regularization = l1_l2_regularization(lamda, l1_ratio)
        else:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0


        super(PolynomialRegression, self).__init__(n_iterations, learning_rate)

    def fit(self, X: np.array, y: np.array):
        X = polynomial_features(X, self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X: np.array) -> np.array:
        X = polynomial_features(X, self.degree)
        return super(PolynomialRegression, self).predict(X)