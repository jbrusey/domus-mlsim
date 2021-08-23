"""
process CWT data into a form suitable for machine learning.

J. Brusey
"""

import numpy as np
import pandas as pd


def df_to_xy(df, x_columns, u_columns, xlag=None, ulag=None):

    """given a dataframe df and x_columns, u_columns being distinct list of
    columns, return dataframe in xy form (x is input, y is output).

    Parameters
    ----------

    df : pandas dataframe

      time series dataframe

    x_columns : list of strings

      columns to be used at input (time t) and output (time t+1)

    u_columns : list of strings

      list of columns in the dataframe that make up the control or $u$ vector

    xlag : int

      number of time steps of state $x$ to include

    ulag : int

      number of time steps of control $u$ to include

    Returns
    -------

    x, y : pandas dataframes
      x (input) and y (output)

    """

    # create shifted versions x1_, x2_, ... so that if x1_ is for time step t, then x2_ is for t+1
    #
    # the smallest shift must be zero - which will be named xn_ where n is the xlag + 1 value

    xdf = [df[x_columns].copy().shift(xlag - i) for i in range(xlag + 1)]
    for i in range(xlag + 1):
        xdf[i].columns = f"x{i + 1}_" + xdf[i].columns

    # as with x, u is shifted u1_, u2_, ... so that un_ where n is ulag matches with xn

    udf = [df[u_columns].copy().shift(ulag - 1 - i) for i in range(ulag)]
    for i in range(ulag):
        udf[i].columns = f"u{i + 1}_" + udf[i].columns

    vdf = []
    rows_to_drop = max(xlag, ulag)

    x = pd.concat(xdf[:-1] + vdf + udf, axis=1).iloc[rows_to_drop:]
    y = xdf[-1].iloc[rows_to_drop:]
    return (x, y)


def unroll_by_group(df, group_column, x_columns, u_columns, xlag, ulag):

    """given a single dataframe that has a column indicating the
    group, form unrolled rows X=x1,x2,...,u1,u2,... Y=xn, similar
    to df_to_train_test but without splitting into train / test. This
    can then be used with sklearn cross-validation methods.

    Parameters
    ----------

    df : pandas dataframe

      data frame containing time series data

    group_column : str

      name of column containing group number

    x_columns : list of strings

      list of columns in the dataframe that make up the state or $x$ vector

    u_columns : list of strings

      list of columns in the dataframe that make up the control or $u$ vector

    xlag : int

      number of time steps of state $x$ to include

    ulag : int

      number of time steps of control $u$ to include

    Returns
    -------

    X, y, group: pandas frame x 2, np.array

      pandas frames where X contains input data, y contains output
      data, group contains one element per row in X (or y) indicating
      the group it belongs to.

    """

    assert isinstance(group_column, str)
    assert isinstance(df, pd.DataFrame)
    groups = df[group_column].unique()

    xy = [
        df_to_xy(
            df[df[group_column] == exp], x_columns, u_columns, xlag=xlag, ulag=ulag
        )
        for exp in groups
    ]

    # generate group column
    glist = np.repeat(groups, repeats=[len(x) for x, y in xy])

    X, y = [pd.concat(x, ignore_index=True) for x in list(zip(*xy))]
    return X, y, glist
