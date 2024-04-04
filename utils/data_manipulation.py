import numpy as np
from typing import Tuple
from itertools import combinations_with_replacement


def shuffle_data(X: np.array,y: np.array, seed=None) -> Tuple[np.array,np.array]:
    """
    Shuffles arrays X and y

    Parameters
    -------
    X: first array
    y: second array
    seed: seed for the shuffling

    Returns
    -------
    return the shuffled arrays
    """
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def batch_iterator(X: np.array,y: np.array = None, batch_size=10):
    """
    Functions that returns batches of the data

    Parameters
    -------
    X: first array
    y: second array (optional)
    batch_size: ammount of batches we generate (default is 10)
    
    Returns
    -------
    yields the batches that are generated
    """
    for i in np.arange(0, X.shape[0], batch_size):
        begin, end = i, min(i + batch_size, X.shape[0])
        if y is not None:
            yield X[begin:end],y[begin:end]
        else:
            yield X[begin:end]

def split_on_feature(X: np.array, feature: int, threshold) -> Tuple[np.array, np.array]:
    """
    Function that splits the dataset into two based on a threshold

    Parameters
    -------
    X: dataset
    feature: index of feature that we split on
    threshold: threshold of feature

    Returns
    -------
    Tuple containg the two splits
    """

    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature] > threshold
    else:
        split_func = lambda sample: sample[feature] == threshold 

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return X_1, X_2


def normalize_data(X: np.array, axis: int = -1, order: int = 2) -> np.array:
    """
    normalizes the data

    Parameters
    -------
    X: dataset to normalize
    axis: axis to normalize on (default is the last axis)
    order: order of norm (default is l2 norm)
    """

    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1 # so if we have l2 norm equal to 0 we dont get error
    return X / np.expand_dims(l2, axis)

def std_scale(X: np.array) -> np.array:
    """
    scales dataset using standardization

    Parameters
    -------
    X: dataset to standardize

    Returns
    -------
    X_std: Standardized dataset
    """

    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    for col in range(np.shape(X)[1]):
        X_std[:,col] = (X[:,col] - mean[col]) / std[col]

    return X_std

def min_max_scale(X: np.array, min: float=-1., max: float=1.) -> np.array:
    """
    Scales data between -1 and 1

    Parameters
    -------
    X: dataset to scale
    min: min value of scale
    max: max value of scale

    Returns
    -------
    X_scaled: scaled dataset
    """
    min_X = np.min(X)
    max_X = np.max(X)

    X_scaled  = (X - min_X) / (max_X - min_X) * (max - min) + min
    return X_scaled

def make_diagonal(X: np.array) -> np.array:
    """
    Converst dataset to diagonal array

    Parameters:
    -------
    X: data array

    Returns:
    ------- 
    X_diag: diagonal array of X
    """ 

    X_diag = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        X_diag[i,i] = X[i,i]
    
def one_hot_encode(X: np.array) -> np.array:
    """
    Encodes X using one hot encoding

    Parameters
    -------
    X: feature to perfrom OneHotEncoding
    
    Returns
    -------
    one_hot_x: X encoding using OneHotEncoding
    """

    if X.ndim != 1:
        return None

    one_hot_encode = []
    unique_values = np.unique(X)
    for element in X:
        arr = np.zeros(len(unique_values))
        arr[np.where(unique_values == element)] = 1
        one_hot_encode.append(arr)
    return np.array(one_hot_encode)

def train_test_split(X: np.array,y: np.array,test_size: float=0.5,seed=None):
    """
    splits X,y into training & testing sets
    
    Parameters:
    -------
    X: dataset
    y: labels
    seed: seed for shuffling 
    
    Returns
    -------
    returns tuple with (X_train,X_test,y_train,y_test)
    """

    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train,X_test = X[:split_i],X[split_i:]
    y_train,y_test = y[:split_i],y[split_i:]
    return X_train,X_test,y_train,y_test

def polynomial_features(X: np.array, degree: int):
    """
    Function for generating polynomial features from dataset

    Parameters:
    -------
    X: dataset
    degree: polynomial degree
    """

    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new