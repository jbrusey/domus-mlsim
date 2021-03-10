from timeseriesgroup import TimeSeriesGroupSplit
import numpy as np

from numpy.testing import assert_array_equal


def test_timeseriesgroup():

    tscv = TimeSeriesGroupSplit(n_splits=3)  # test size = 5
    X = np.zeros((20, 1))
    groups = np.array([1] * 7 + [2] * 13)

    i = 0
    for train, test in tscv.split(X, groups=groups):
        if i == 0:
            assert_array_equal(train, np.array([0, 1, 2, 3, 7, 8, 9, 10]))
            assert_array_equal(test, np.array([4, 11, 12, 13]))
        elif i == 1:
            assert_array_equal(train, np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]))
            assert_array_equal(test, np.array([5, 14, 15, 16]))
        elif i == 2:
            assert_array_equal(train, np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
            assert_array_equal(test, np.array([6, 17, 18, 19]))
        i += 1
