import numpy as np
from utils.data_operations import calc_euclidean_distance

class KNN:
    """
    Parameters
    -------
    k: number of neighbours that will determine the class of the instances
    """
    def __init__(self, k: int):
        self.k = k

    def fit(self, X, y): pass

    """
    function that returns the label of the most common neighbors

    Parameters
    -------
    neighbor_labels: labels of the k closesneighbors
    """
    def _vote(self, neighbor_labels: np.array) -> int:
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: np.array) -> np.array:
        y_pred = np.empty(X_test.shape[0])
        for i, train_instance in enumerate(X_test):
            idx = np.argsort([calc_euclidean_distance(train_instance, x) for x in self.X_train])[:self.k]
            k_nearest_neighbors = np.array([self.y_train[i] for i in idx])
            y_pred[i] =  self._vote(k_nearest_neighbors)
        return np.array(y_pred)