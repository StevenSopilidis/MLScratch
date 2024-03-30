from supervised_learning.perceptron import Perceptron
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from utils.data_manipulation import std_scale, normalize_data, train_test_split, one_hot_encode
from utils.data_operations import accuracy_score
import pandas as pd
import numpy as np
from sklearn import datasets

data = datasets.load_digits()
X = normalize_data(data.data)
y = data.target

# One-hot encoding of nominal y-values
y = one_hot_encode(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)
clf = Perceptron(n_iterations=5000,
    learning_rate=0.001)
clf.fit(X_train, y_train)
y_pred = np.argmax(clf.predict(X_test), axis=1)
y_test = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print ("Accuracy:", accuracy)