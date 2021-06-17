from typing import Union

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class TimeWindowTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, columns: list, rolling=1, aggregate: Union[str, dict] = 'sum', dropna=False) -> None:
        self.columns = columns
        self.rolling = rolling
        self.aggregate = aggregate
        self.dropna = dropna

    def fit(self, X, y=None) -> 'TimeWindowTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_
            __rolling__ = None
            __dropna__ = False

        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan', ensure_2d=False)

        self.is_fitted_ = True

        __rolling__ = self.rolling
        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        new_X = X.copy() if copy else X
        temp = X[self.columns].rolling(self.rolling).agg(self.aggregate)

        for col in self.columns:
            new_X[col] = temp[col]

        del temp
        return new_X.dropna() if self.dropna else new_X
