
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(DataFrame(X))
        print(X.shape)
        return X

    def fit(self, X, y=None, **fit_params):
        return self
