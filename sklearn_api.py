# coding: utf-8
from functools import partial
from itertools import product

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.model_selection import StratifiedKFold

from sklearn.utils import check_X_y


class KNeighborsFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_neighbors=1,
        n_splits=5,
        shuffle=False,
        random_state=None
    ):
        self.n_neighbors = n_neighbors
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    @staticmethod
    def __distance(v, u):
        return np.linalg.norm(v - u)

    def _extract_feature_value(self, vector, X, y, class_label, k):
        X_target_class = X[y == class_label]
        euclidean_vec = partial(self.__distance, vector)
        distances = np.array([
            euclidean_vec(vector) for vector in X_target_class])
        return np.sum(np.sort(distances)[:k + 1])

    def _extract_feature_vectors(self, matrix, X, y, class_label, k):
        _extract_feature_value = partial(
            self._extract_feature_value,
            X=X, y=y, class_label=class_label, k=k)
        return np.array([
            _extract_feature_value(vector) for vector in matrix])

    def transform(self, X, y):
        X, y = check_X_y(X, y, force_all_finite=True)

        class_labels = set(y)
        row_size = np.array(X).shape[0]
        X_knn = np.empty((row_size, len(class_labels) * self.n_neighbors))
        skf = StratifiedKFold(
            self.n_splits, self.shuffle, self.random_state)

        for train_idx, test_idx in skf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]
            features = [
                self._extract_feature_vectors(
                    X_test, X_train, y_train, class_label, k)
                for class_label, k in product(
                    class_labels, range(self.n_neighbors))
            ]
            X_knn[test_idx] = np.array(features).T
        return X_knn
