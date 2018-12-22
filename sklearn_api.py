# coding: utf-8
from functools import partial
from itertools import product

from faiss import IndexFlatL2

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.neighbors import NearestNeighbors

from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_random_state


class KNeighborsFeatures(BaseEstimator, TransformerMixin):
    allowed_methods = {
        'sklearn': NearestNeighbors,
        'faiss': IndexFlatL2,
    }

    def __init__(
        self,
        n_neighbors=1,
        sampling_ratio=0.8,
        method='sklearn',
        mean=0,
        sigma=0.1,
        n_jobs=1,
        random_state=None,
        verbose=0,
        **kwargs
    ):
        self.n_neighbors = n_neighbors
        if not isinstance(sampling_ratio, dict) \
           and not isinstance(sampling_ratio, float):
            raise ValueError(
                'sampling_ratio must be dict or float. {}'.format(
                    sampling_ratio
                )
            )
        if method not in KNeighborsFeatures.allowed_methods:
            raise ValueError('"{}" is not supported.'.format(method))
        self.method = method
        self.sampling_ratio = sampling_ratio
        self.mean = mean
        self.sigma = sigma
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.kwargs = kwargs

    def _search(self, X):
        def __search_func(indexer):
            if 'sklearn' == self.method:
                dists, _ = indexer.kneighbors(X, self.n_neighbors)
            elif 'faiss' == self.method:
                dists, _ = indexer.search(X, k=self.n_neighbors)
            return dists

        return {
            class_label: __search_func(idxer)
            for class_label, idxer in self._X_each_class.items()
        }

    def _extract_feature(self, X):
        dim = self.n_neighbors * len(self._X_each_class)
        neighbors_dists = self._search(X)
        X_knn = np.empty((X.shape[0], dim))
        for idx, (k, class_label) in enumerate(
            product(range(self.n_neighbors), self._X_each_class.keys())
        ):
            X_knn[:, idx:idx + 1] = np.sum(
                neighbors_dists[class_label][:, 0:k + 1],
                axis=1,
                keepdims=True
            )
        return X_knn

    def __get_indexer_initializer(self):
        search_klass = KNeighborsFeatures.allowed_methods[self.method]
        return partial(search_klass, **self.kwargs)

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
        return np.array(X + noize.reshape(X.shape), dtype=np.float32)

    def _train(self, func, X):
        if 'sklearn' == self.method:
            return func(n_jobs=self.n_jobs).fit(X)
        elif 'faiss' == self.method:
            indexer = func(X.shape[1])
            indexer.add(X)
            return indexer

    def fit(self, X, y):
        X, y = check_X_y(X, y, force_all_finite=True)
        init = self.__get_indexer_initializer()
        class_set = set(y)
        self.__set_sampling_ratio_each_class(class_set)
        self._X_each_class = {}
        for class_label in class_set:
            noize_added_X = self.__add_gaussian_noize(X[y == class_label])
            sampled_X = self.__choice(
                noize_added_X,
                self._sampling_ratio_each_class[class_label]
            )
            self._X_each_class[class_label] = self._train(init, sampled_X)
        return self

    def transform(self, X):
        check_is_fitted(self, '_X_each_class')
        X = np.array(check_array(X, force_all_finite=True), dtype=np.float32)
        X_knn = self._extract_feature(X)
        return X_knn
