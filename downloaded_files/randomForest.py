# author: Jan Kwiatkowski

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

from model.decisionTree_id3 import ID3Tree


class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trees=25, max_depth=100, min_samples_split=2, random_state=None):

        # hiperparametry drzewa ID3
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        # hiperparametry lasu
        self.n_trees = n_trees
        self.random_state = random_state
        self.trees = []

    def sample(self, X, y, random_state):
        n_rows, n_cols = X.shape

        np.random.seed(random_state)
        samples = np.random.choice(a=n_rows, size=n_rows, replace=True)

        return X[samples], y[samples]

    def fit(self, X, y):
        # Reset
        if len(self.trees) > 0:
            self.trees = []

        num_built = 0

        while num_built < self.n_trees:
            clf_id3 = ID3Tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            _X, _y = self.sample(X, y, self.random_state)

            clf_id3.fit(_X, _y)
            self.trees.append(clf_id3)
            num_built += 1

    def predict(self, X):
        # Predykcja wyznaczana dla kazdego klasyfikatora w lesie
        y = []
        for tree in self.trees:
            y.append(tree.predict(X))

        y = np.swapaxes(y, axis1=0, axis2=1)

        # Glosowanie wiekszosciowe
        predicted_classes = stats.mode(y, axis=1, keepdims=True)[0].reshape(-1)

        return predicted_classes
