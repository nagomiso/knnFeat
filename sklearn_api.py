# coding: utf-8
from functools import partial

from joblib import delayed
from joblib import Parallel

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.utils import check_X_y
from sklearn.utils import chech_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_random_state


class KNeighborsFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_neighbors=1,
        sampling_ratio=0.8,
        mean=0,
        sigma=0.1,
        n_jobs=1,
        random_state=None,
        verbose=0
    ):
        self.n_neighbors = n_neighbors
        self.sampling_ratio = sampling_ratio
        self.mean = mean
        self.sigma = sigma
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        self.verbose = verbose

    def _extract_feature_values(self, x):
        calc_dist = partial(
            KNeighborsFeatures.__distance, x)
        x_knn = []
        feature_append = x_knn.append
        for X_train in self._X_each_class.values():
            distances = np.sort(
                np.array([calc_dist(x_train) for x_train in X_train]))
            for k in range(self.n_neighbors):
                feature_append(np.sum(distances[:k + 1]))
        return np.array(x_knn)

    @staticmethod
    def __distance(v, u):
        return np.linalg.norm(v - u)

    def __choice(self, X, sampling_ratio):
        if sampling_ratio == 1:
            return X
        sample_size = int(X.shape[0] * sampling_ratio)
        idx = self.random_state.choice(
            X.shape[0], sample_size, replace=False)
        return X[np.sort(idx), :]

    def __set_sampling_ratio_each_class(self, class_set):
        if isinstance(self.sampling_ratio, float):
            self._sampling_ratio_each_class = {
                class_label: self.sampling_ratio
                for class_label in class_set
            }
        elif isinstance(self.sampling_ratio, dict):
            if set(self.sampling_ratio.keys()).issubset(class_set):
                self._sampling_ratio_each_class = self.sampling_ratio

    def __add_gaussian_noize(self, X):
        noize = self.random_state.normal(
            self.mean, self.sigma, np.prod(X.shape))
        return X + noize.reshape(X.shape)

    def fit(self, X, y):
        X, y = check_X_y(X, y, force_all_finite=True)
        self._dim = X.shape[1]
        class_set = set(y)
        self.__set_sampling_ratio_each_class(class_set)
        self._X_each_class = {
            class_label: self.__choice(
                self.__add_gaussian_noize(X[y == class_label]),
                self._sampling_ratio_each_class[class_label]
            )
            for class_label in class_set
        }
        return self

    def transform(self, X):
        check_is_fitted(self, '_X_each_class')
        X_knn = np.array(
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                [delayed(self._extract_feature_values)(x) for x in X])
            )
        return X_knn
