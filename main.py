from Supervised_learning.logistic_regression import LogisticRegression
from sklearn.datasets import load_iris
from Utils.data_manipulation import std_scale, normalize_data
from Utils.data_operations import accuracy_score
import pandas as pd

data = load_iris()
X = std_scale(data.data[data.target != 0])
y = data.target[data.target != 0]
y[y == 1] = 0
y[y == 2] = 1

clf = LogisticRegression()
clf.fit(X, y)
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print ("Accuracy:", accuracy)