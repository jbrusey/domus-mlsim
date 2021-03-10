"""

TimeSeriesGroupSplit

Author
------
J. Brusey

"""

from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
import numpy as np


class TimeSeriesGroupSplit(BaseCrossValidator):

    def __init__(self,
                 n_splits=5,
                 ):
        super().__init__()
        self.n_splits = int(n_splits)

    def get_n_splits(self):
        return self.n_splits
        
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Required

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """

        unique_groups = np.unique(groups)
        n_splits = self.n_splits
        indices = np.arange(len(groups))
        g_indices = [indices[groups == g] for g in unique_groups]
        splits = [TimeSeriesSplit(n_splits).split(g_indices[i]) for i in range(len(g_indices))]

        for i in range(self.n_splits):
            train = []
            test = []
            for j, g_index in enumerate(g_indices):
                train_i, test_i = next(splits[j])
                train.append(g_index[train_i])
                test.append(g_index[test_i])

            yield np.hstack(train), np.hstack(test)
