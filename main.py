# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from utils.data_manipulation import train_test_split
from supervised_learning.knn import KNN
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize KNN classifier
# knn = KNeighborsClassifier(n_neighbors=3)  # K=3, you can change it according to your preference

# Train the classifier
# knn.fit(X_train, y_train)

# Predict on the test data
# y_pred = knn.predict(X_test)

knn = KNN(k = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
