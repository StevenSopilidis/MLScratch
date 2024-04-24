# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from utils.data_operations import accuracy_score
from utils.data_manipulation import one_hot_encode, normalize_data, train_test_split
from supervised_learning.ada_boost import AdaBoost
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn import datasets
from unsupervised_learning.pca import PCA
import numpy.random as rnd



mu = np.array([10,13])
sigma = np.array([[3.5, -1.8], [-1.8,3.5]])

# print("Mu ", mu.shape)
# print("Sigma ", sigma.shape)

# Create 1000 samples using mean and sigma
org_data = rnd.multivariate_normal(mu, sigma, size=(1000))
# print("Data shape ", org_data.shape)

pca = PCA(1)
projected = pca.fit(org_data)
print("Compressed Data shape ", projected.shape)
original = pca.inverse_transform(projected)
print("Compressed Data shape ", original.shape)
