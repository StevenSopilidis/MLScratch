from supervised_learning.perceptron import Perceptron
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from utils.data_manipulation import std_scale, normalize_data
from utils.data_operations import accuracy_score
import pandas as pd

data = load_iris()
X = std_scale(data.data[data.target != 0])
y = data.target[data.target != 0]
y[y == 1] = 0
y[y == 2] = 1

clf = Perceptron()
clf.fit(X, y)
y_pred = clf.predict(X)
print(y_pred)