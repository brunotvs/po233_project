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

        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan', ensure_2d=False)

        self.is_fitted_ = True

        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        new_X = X if copy == False else X.copy()
        temp = X[self.columns].rolling(self.rolling).agg(self.aggregate)

        for col in self.columns:
            new_X[col] = temp[col]

        del temp
        return new_X.dropna() if self.dropna else new_X


class VarShiftTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, columns: list, shift=0, dropna=False) -> None:
        self.columns = columns
        self.shift = shift
        self.dropna = dropna

    def fit(self, X, y=None) -> 'VarShiftTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan', ensure_2d=False)

        self.is_fitted_ = True

        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        new_X = X if copy == False else X.copy()
        temp = X[self.columns].rolling(self.rolling).agg(self.aggregate)

        for col in self.columns:
            new_X[col] = temp[col]

        del temp
        return new_X.dropna() if self.dropna else new_X
