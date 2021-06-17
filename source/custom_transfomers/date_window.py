from typing import Union

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


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

        new_X = X.copy()
        temp = X[self.columns].rolling(self.rolling).agg(self.aggregate)

        for col in self.columns:
            new_X[col] = temp[col]

        del temp
        return new_X.dropna() if self.dropna else new_X


class YTimeWindowTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, x_time_windowing_transformer: TimeWindowTransformer) -> None:
        self.x_time_windowing_transformer = x_time_windowing_transformer

    def fit(self, X, y=None) -> 'TimeWindowTransformer':

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_
            del self.rolling

        X = check_array(X, ensure_2d=True)

        self.is_fitted_ = True

        return self

    def transform(self, X: DataFrame, copy=None) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        new_X = X.copy() if copy else None
        if self.x_time_windowing_transformer.dropna:
            new_X.drop(range(self.x_time_windowing_transformer.rolling))

        return new_X

    def inverse_transform(self, X: DataFrame, copy=None) -> DataFrame:
        new_X = X.copy() if copy else X
        return new_X


class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(DataFrame(X))
        print(X.shape)
        return X

    def fit(self, X, y=None, **fit_params):
        return self


class TemplateTransformer(TransformerMixin, BaseEstimator):
    """ An example transformer that returns the element-wise square root.
    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return np.sqrt(X)
