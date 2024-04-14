# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from utils.data_operations import accuracy_score
from utils.data_manipulation import one_hot_encode, normalize_data, train_test_split
from supervised_learning.mlp import MLP
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn import datasets


data = datasets.load_digits()
X = normalize_data(data.data)
y = data.target

# Convert the nominal y values to binary
y = one_hot_encode(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

# MLP
clf = MLP(n_hidden=16,
    n_iterations=1000,
    learning_rate=0.01)

clf.fit(X_train, y_train)
y_pred = np.argmax(clf.predict(X_test), axis=1)
y_test = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print ("Accuracy:", accuracy)