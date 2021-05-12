"""
PartialScaler


Author
------
J. Brusey, 4-March-2021


Scaler wrapper to support separating out the scaling of a single
vector into two separate x and u vectors.

"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class PartialScaler(TransformerMixin, BaseEstimator):

    def __init__(self, scaler, lower, upper, n_features):
        """.

        Wrapper for an ordinary scaler object to make it one that will
        scale a partially filled vector. For example, if the original
        scaler is over N features and you want a new scaler that just
        scales over the first k features and another scaler for the
        rest, you would use

          px = PartialScaler(scaler, 0, k, N)
          py = PartialScaler(scaler, k, N, N)

        Parameters
        ----------

        scaler : sklearn scaler object

          original scaler that scales over all n_features

        lower : integer

          index to lower bound of original vector to create new scaler form

        upper : integer

          upper bound (following usual python slice semantics where upper bound index is not included)

        n_features : integer

          number of features in the original scaler

        """
        self.scaler = scaler
        self.lower = lower
        self.upper = upper
        self.n_features = n_features

    def fit(self, X, y=None):
        raise NotImplementedError("fit wrapped scaler instead")

    def zeropad(self, X):
        assert X.shape[1] == self.upper - self.lower
        return np.hstack([np.zeros((X.shape[0], self.lower)),
                          X,
                          np.zeros((X.shape[0], self.n_features - self.upper))])

    def strippad(self, X):
        return X[:, self.lower:self.upper]

    def transform(self, X):
        return self.strippad(
            self.scaler.transform(
                self.zeropad(X)))

    def inverse_transform(self, X):
        return self.strippad(
            self.scaler.inverse_transform(
                self.zeropad(X)))
