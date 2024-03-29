import numpy as np
from sklearn import datasets
import math

class NaiveBayes():
    """
    Naive Bayes classifier
    """

    def fit(self, X: np.array, y: np.array):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.parameters = []
        # calculate μ,σ for each feature for each class
        for i,c in enumerate(self.classes):
            # get all instances of dataset that belong to class c
            X_where_c = X[np.where(self.y == c)]
            self.parameters.append([])
            for col in X_where_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)


    def _calculate_prior(self, c) -> float:
        """
        Calculates prior propability of class c

        Parameters
        -------
        c: class to calcuate prior probability

        Returns
        -------
        prior: prior propability of class c
        """

        prior = len(self.X[np.where(self.y == c)]) / len(self.X)
        return prior
    
    def _calculate_likelyhood(self, mean: float, var: float, x: np.array) -> float:
        """
        Function for calculating likelyhood using Normal distribution func

        Paramaters
        -------
        mean: mean of the class
        var: variance of the class
        x: data of class c to calculate likelyhood on

        Returns
        -------
        likelyhood: likelyhood of class c
        """
        coeff = 1 / var * math.sqrt(2*math.pi)
        exp = math.exp(-math.pow(x-mean,2)/(2*var))
        likelyhood = coeff * exp
        return likelyhood
    
    def _classify(self, sample: np.array):
        """
        Function that classifies data point based on bayes rule

        Parameters
        -------
        sample: data point to classify

        Returns
        -------
        class: class the sample belongs to
        """
        posteriors = []
        """Naive assumption: independance of features"""
        for i, c in enumerate(self.classes):
            posterior = self._calculate_prior(c)
            for feature_value, params in zip(sample, self.parameters[i]):
                likelyhood = self._calculate_likelyhood(params["mean"], params["var"], feature_value)
                posterior *= likelyhood
            posteriors.append(posterior)
        """Return the class with the highest posterio propability"""
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X: np.array) -> np.array:
        return np.array([self._classify(sample) for sample in X])