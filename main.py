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


data = datasets.load_digits()
X = data.data
y = data.target

digit1 = 1
digit2 = 8
idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
y = data.target[idx]
# Change labels to {-1, 1}
y[y == digit1] = -1
y[y == digit2] = 1
X = data.data[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


# Adaboost classification with 5 weak classifiers
clf = AdaBoost(n_clfs=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print ("Accuracy:", accuracy)
