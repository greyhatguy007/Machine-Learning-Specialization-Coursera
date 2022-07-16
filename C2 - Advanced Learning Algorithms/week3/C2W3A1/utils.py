import numpy as np
from sklearn import datasets


def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    X = X[y != 2] # only two classes
    y = y[y != 2]
    return X, y