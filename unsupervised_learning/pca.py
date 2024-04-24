import numpy as np

class PCA:
    """
    Class that implements pca dimensionality reduction
    
    Parameters
    -------
    dims: number of dimensions to of projected data
    """
    def __init__(self, dims) -> None:
        self.dims = dims

    def fit(self, X: np.array) -> np.array:
        n_samples, n_features = np.shape(X)

        if n_features < self.dims:
            raise ValueError("Dimensions of projected data must be lower or equal to initial diminsions")
        
        # center data
        mean = np.mean(X, axis=0)
        self.mean = mean
        mean_data = X - mean

        # calculate covariance matrix 
        cov = np.cov(mean_data.T)

        # get the eigen_values && eigen_vectors of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eig(cov)

        # sort the eigen_vectors using the eigen_values (from max to min)
        indices = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[indices]
        eigen_vectors = eigen_vectors[:, indices]
        self.eigen_vectors = eigen_vectors


        # calculate the explained_variance
        self.explained_variance = eigen_values / np.sum(eigen_values)

        self.cumulative_variance = np.cumsum(self.explained_variance)

        projected_data = np.dot(mean_data, eigen_vectors[:, :self.dims])

        # project data
        return projected_data

    def inverse_transform(self, y: np.array) -> np.array:
        return np.dot(y, self.eigen_vectors[:, :self.dims].T) + self.mean
