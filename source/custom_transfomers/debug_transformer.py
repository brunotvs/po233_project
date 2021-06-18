
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(DataFrame(X))
        # print(DataFrame(X).info())
        print(X.shape)
        return X

    def fit(self, X, y=None, **fit_params):
        return self

    def inverse_transform(self, X):
        return X

    def predict(self, X):
        return X


class DebugY(BaseEstimator, TransformerMixin):

    def transform(self, X):
        return X

    def fit(self, X, y=None, **fit_params):
        print(y)
        return self

    def inverse_transform(self, X):
        return X

    def predict(self, X):
        return X
