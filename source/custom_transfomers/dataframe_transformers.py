from typing import Any, Set, Type, Union

import numpy as np
from pandas import DataFrame, MultiIndex, Index
import pandas
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from source.project_utils.data_manipulation import ColumnsLoc


class PivotTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, index=None, columns=None, values=None) -> None:
        self.index = index
        self.columns = columns
        self.values = values

    def fit(self, X, y=None) -> 'PivotTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        self.is_fitted_ = True

        self.transform(y)

        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        new_X = X

        new_X.pivot(
            index=self.index,
            columns=self.columns,
            values=self.values
        )

        return new_X


class ResetIndexTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, level=None, drop=False, inplace=True, col_level=0, col_fill='') -> None:
        self.level = level
        self.drop = drop
        self.inplace = inplace
        self.col_level = col_level
        self.col_fill = col_fill

    def fit(self, X, y=None) -> 'ResetIndexTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        self.is_fitted_ = True

        self.transform(y)

        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        temp_X = X.reset_index(
            level=self.level,
            drop=self.drop,
            inplace=self.inplace,
            col_level=self.col_level,
            col_fill=self.col_fill
        )

        X = temp_X

        return X


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

        self.is_fitted_ = True

        self.transform(y)
        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        temp_df = X.groupby(
            by=self.by,
            axis=self.axis,
            level=self.level,
            as_index=self.as_index,
            sort=self.sort,
            group_keys=self.group_keys,
            squeeze=self.squeeze,
            observed=self.observed,
            dropna=self.dropna
        )

        X = temp_df

        return X


class AggregateTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, func=None, axis=0, *args, **kwargs) -> None:
        self.func = func
        self.axis = axis
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y=None) -> 'PivotTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        self.is_fitted_ = True

        self.transform(y)

        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        temp_X = X.agg(
            func=self.func,
            axis=self.axis,
            args=self.args,
            kwargs=self.kwargs
        )

        X = temp_X

        return X


class TestTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, n=0) -> None:
        self.n = n

    def fit(self, X: DataFrame, y=None) -> 'TestTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        self.is_fitted_ = True

        self.transform(y)

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        X.drop(X.index[:self.n], inplace=True)

        return X

    def inverse_transform(self, X):
        return X


class TestTargetColumnSelector(BaseEstimator, RegressorMixin):

    def __init__(self, column: str = None) -> None:
        self.column = column

    def fit(self, X: DataFrame, y: DataFrame = None) -> 'TestTargetColumnSelector':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        self.is_fitted_ = True

        self._remove_others(y, self.column)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        return X

    def inverse_transform(self, X):
        return X

    @staticmethod
    def _remove_others(df: DataFrame, column: str):

        diff = set()
        if isinstance(df.columns, MultiIndex):
            diff = TestTargetColumnSelector._diff_from_multiindex(df.columns, column)

        elif isinstance(df.columns, Index):
            diff = TestTargetColumnSelector._diff_from_index(df.columns, column)

        else:
            raise TypeError(f"columns must be Index or MultiIndex, was {type(df.columns)} instead")

        df.drop(diff, axis=1, inplace=True)

    @staticmethod
    def _diff_from_multiindex(columns: MultiIndex, column: str):
        cols_total = {col for col in columns}
        column_set = set()
        for col in cols_total:
            if column == col[0]:
                column_set.add(col)

        diff: Set[Any] = cols_total - column_set
        return diff

    @staticmethod
    def _diff_from_index(columns: Index, column: str):
        column_set = set(column)
        cols_total: Set[Any] = set(columns)
        diff: Set[Any] = cols_total - column_set
        return diff


def _clear_dataframe(df: DataFrame):
    df.drop(df.columns, 1)
