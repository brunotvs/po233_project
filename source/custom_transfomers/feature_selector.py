from typing import Union

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class DFColumnSelector(TransformerMixin, BaseEstimator):
    """Object for selecting specific columns from a data set.

    Parameters
    ----------
    cols : array-like (default: None)
        A list specifying the feature indices to be selected. For example,
        [1, 4, 5] to select the 2nd, 5th, and 6th feature columns, and
        ['A','C','D'] to select the name of feature columns A, C and D.
        If None, returns all columns in the array.

    drop_axis : bool (default=False)
        Drops last axis if True and the only one column is selected. This
        is useful, e.g., when the ColumnSelector is used for selecting
        only one column and the resulting array should be fed to e.g.,
        a scikit-learn column selector. E.g., instead of returning an
        array with shape (n_samples, 1), drop_axis=True will return an
        aray with shape (n_samples,).

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/ColumnSelector/

    """

    def __init__(self, cols=None, drop_axis=False):
        self.cols = cols
        self.drop_axis = drop_axis

    def fit(self, X, y=None):

        if hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan', ensure_2d=False)

        self.is_fitted_ = True

        return self

    def transform(self, X, y=None):

        # We use the loc or iloc accessor if the input is a pandas dataframe
        if hasattr(X, 'loc') or hasattr(X, 'iloc'):
            if isinstance(self.cols, tuple):
                self.cols = list(self.cols)
            types = {type(i) for i in self.cols}
            if len(types) > 1:
                raise ValueError(
                    'Elements in `cols` should be all of the same data type.'
                )
            if isinstance(self.cols[0], int):
                t = X.iloc[:, self.cols].values
            elif isinstance(self.cols[0], str):
                t = X.loc[:, self.cols].values
            else:
                raise ValueError(
                    'Elements in `cols` should be either `int` or `str`.'
                )
        else:
            t = X[:, self.cols]

        if t.shape[-1] == 1 and self.drop_axis:
            t = t.reshape(-1)
        if len(t.shape) == 1 and not self.drop_axis:
            t = t[:, np.newaxis]
        return t
