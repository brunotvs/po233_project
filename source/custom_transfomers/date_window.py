from typing import Union

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
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

        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan')

        self.is_fitted_ = True

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        check_is_fitted(self, 'is_fitted_')

        X = X.copy()
        new_X = X.copy()
        X = X[self.columns].rolling(self.rolling).agg(self.aggregate)

        for col in self.columns:
            new_X[col] = X[col]

        return new_X.dropna() if self.dropna else new_X


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
