import numpy as np
import math
from sklearn.model_selection import cross_val_predict

def calc_entropy(y: np.array) -> float:
    """ 
    Calculate entropy of label array y 

    Parameters
    -------
    y: array of labels

    Returns
    -------
    entropy: entropy of given labels
    """

    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = len(y)/count
        entropy += -p*math.log2(p)
    return entropy

def calc_mse(y_pred: np.array, y_labels: np.array) -> float:
    """
    Calculates the MSE of the predicted values

    Parameters
    -------
    y_pred: predicted values
    y_labels: actual values
    
    Returns
    -------
    mse of predictions
    """

    return np.mean(np.power(y_labels - y_pred, 2))

def calc_variance(X: np.array) -> np.array:
    """
    Calculates variance of the features of a dataset

    Parameters
    -------
    X: dataset

    Returns
    -------
    variance of features in dataset X
    """

    means = np.mean(X, axis=0)
    n_samples = np.shape(X)[0]
    return (1/n_samples) * np.diag((X - means)).T.dot((X - means))


def calc_std_deviation(X: np.array) -> np.array:
    """
    Calculates standard deviation of dataset

    Parameters
    -------
    X: dataset

    Returns
    -------
    standard deviation of dataset X
    """

    return math.sqrt(calc_variance(X))

def calc_euclidean_distance(x1: np.array, x2: np.array) -> float:
    """
    Calculates euclidean distance (l2 norm) between two arrays

    Parameters
    -------

    x1: first array
    x2: second array

    Returns
    -------
    l2 norm of x1 & x2
    """

    distance = 0
    for i in range(len(x1)):
        distance += math.pow((x1[i]-x2[i]), 2)
    
    return math.sqrt(distance)

def calc_covariance_matrix(X: np.array) -> np.array:
    """
    Calculates covariance matrix of dataset

    Parameters
    -------
    X: dataset

    Returns
    -------
    covariance matrix of dataset
    """

    mean = np.mean(X, axis=0)
    centered_data = X - mean
    product = np.dot(centered_data.T, centered_data)
    return product / (X.shape[0] - 1)

def accuracy_score(y_true: np.array, y_pred: np.array) -> float:
    """
    Function that returns accuracy score of predictors

    Parameters
    -------
    y_true: actual labels
    y_pred: predicted labels

    Returns
    -------
    score: accuracy score of model
    """

    score = np.sum(y_true == y_pred, axis=0) / len(y_pred)
    return score