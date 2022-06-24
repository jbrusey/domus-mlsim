"""
util.py

Author
------
J. Brusey

Date
----
May 24, 2021

Description
-----------

various utility functions

"""
import numpy as np


def kw_to_array(columns, dtype=np.float32, **kwargs):
    """convert a set of keywords into a numpy array according to a list
    of column names"""

    assert set(columns) == set(
        kwargs
    ), f"Either additional {set(kwargs) - set(columns)} or missing {set(columns) - set(kwargs)} arguments"
    return np.array([kwargs[c] for c in columns], dtype=dtype)
