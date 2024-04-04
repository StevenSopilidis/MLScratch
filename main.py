# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from utils.data_manipulation import train_test_split
from supervised_learning.regression import PolynomialRegression, Regularization
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

n_samples, n_features = np.shape(X)

model = PolynomialRegression(regularization=Regularization.L2, lamda=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_train)
print(mean_squared_error(y_train, y_pred))