from typing import Any, Set, Type, Union, Tuple

import numpy
from pandas import DataFrame, MultiIndex, Index
import pandas
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sqlalchemy.sql.schema import RETAIN_SCHEMA
from source.project_utils.data_manipulation import ColumnsLoc


class ShapeTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, shape=(-1, 1)) -> None:
        self.shape = shape

    def fit(self, X: numpy.ndarray, y=None):

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        if hasattr(self, 'is_ndarray_'):
            del self.is_ndarray_
            del self.shape_

        self.is_fitted_ = True

        if isinstance(X, numpy.ndarray):
            self.is_ndarray_ = True
            self.shape_ = X.shape

        return self

    def transform(self, X: numpy.ndarray, copy=None) -> numpy.ndarray:
        check_is_fitted(self, 'is_fitted_')

        if self.is_ndarray_:
            if ShapeTransformer._is_oneD(self.shape_):
                return X.reshape(self.shape)

        return X

    @staticmethod
    def _is_oneD(shape: numpy.ndarray):
        return len(shape) == 1
