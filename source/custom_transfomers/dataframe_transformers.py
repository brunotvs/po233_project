from typing import Union

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class PivotTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, index=None, columns=None, values=None) -> None:
        self.index = index
        self.columns = columns
        self.values = values

    def fit(self, X, y=None) -> 'PivotTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan', ensure_2d=False)

        self.is_fitted_ = True

        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        new_X = X if copy == False else X.copy()

        new_X.pivot(
            index=self.index,
            columns=self.columns,
            values=self.values
        )

        return new_X


class ResetIndexTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, level=None, drop=False, inplace=False, col_level=0, col_fill='') -> None:
        self.level = level
        self.drop = drop
        self.inplace = inplace
        self.col_level = col_level
        self.col_fill = col_fill

    def fit(self, X, y=None) -> 'ResetIndexTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan', ensure_2d=False)

        self.is_fitted_ = True

        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        new_X = X if copy == False else X.copy()

        new_X.pivot(
            index=self.index,
            columns=self.columns,
            values=self.values
        )

        return new_X


class GroupByTransformer(TransformerMixin, BaseEstimator):

    def __init__(
            self,
            by=None,
            axis=0,
            level=None,
            as_index=True,
            sort=True,
            group_keys=True,
            squeeze=object,
            observed=False,
            dropna=True) -> None:
        self.by = by
        self.axis = axis
        self.level = level
        self.as_index = as_index
        self.sort = sort
        self.group_keys = group_keys
        self.squeeze = squeeze
        self.observed = observed
        self.dropna = dropna

    def fit(self, X, y=None) -> 'GroupByTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan', ensure_2d=False)

        self.is_fitted_ = True

        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        new_X = X if copy == False else X.copy()

        new_X.groupby(
            by=self.by,
            axis=self.axis,
            level=self.level,
            as_index=self.as_index,
            sort=self.sort,
            group_keys=self.group_keys,
            squeez=self.squeez,
            observed=self.observed,
            dropna=self.dropna
        )

        return new_X


class TestTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, index=None, columns=None, values=None) -> None:
        self.index = index
        self.columns = columns
        self.values = values

    def fit(self, X: DataFrame, y=None) -> 'TestTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan', ensure_2d=False)

        X.drop(X.index[:3], inplace=True)

        self.is_fitted_ = True

        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        new_X = X if copy == False else X.copy()

        new_X.pivot(
            index=self.index,
            columns=self.columns,
            values=self.values
        )

        return new_X
