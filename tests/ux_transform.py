"""
process CWT data into a form suitable for machine learning.

J. Brusey
"""

import pandas as pd
import numpy as np

def xux_rows(df, x_columns, u_columns):
    """given a dataframe df and x_columns and u_columns being distinct
    lists of columns, set up a new dataframe with rows x, u, x+1 where
    x+1 is x shifted by 1

    columns will be renamed x_, u_, x1_

    only columns in x_columns and u_columns are included

    """

    xdf = df[x_columns].copy()
    x1df = df[x_columns].copy()
    udf = df[u_columns].copy()

    xdf.columns = 'x_' + xdf.columns
    udf.columns = 'u_' + udf.columns
    x1df.columns = 'x1_' + x1df.columns

    return (pd.concat([xdf.shift(1), udf.shift(1)], axis=1, ignore_index=True).iloc[1:], x1df.iloc[1:])


def x2ux_rows(df, x_columns, u_columns):
    """ same as xux_rows but shift 2 and provide two states and two controls

    columns will be renamed x1_, x2_, u_, x3_

    """

    x1df = df[x_columns].copy()
    x2df = df[x_columns].copy()
    x3df = df[x_columns].copy()
    udf = df[u_columns].copy()

    udf.columns = 'u_' + udf.columns
    x1df.columns = 'x1_' + x1df.columns
    x2df.columns = 'x2_' + x2df.columns
    x3df.columns = 'x3_' + x3df.columns

    return (pd.concat([x1df.shift(2), x2df.shift(1), udf.shift(1)], axis=1).iloc[2:], x3df.iloc[2:])


def df_to_xy(df, x_columns, u_columns, v_columns=[], xlag=None, ulag=None):

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

    v_columns : list of strings

      list of columns at input (time t-1) but not output

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

    xdf = [df[x_columns].copy().shift(xlag - i)
           for i in range(xlag + 1)]
    for i in range(xlag + 1):
        xdf[i].columns = f'x{i + 1}_' + xdf[i].columns

    # as with x, u is shifted u1_, u2_, ... so that un_ where n is ulag matches with xn

    udf = [df[u_columns].copy().shift(ulag - 1 - i)
           for i in range(ulag)]
    for i in range(ulag):
        udf[i].columns = f'u{i + 1}_' + udf[i].columns

    vdf = []
    rows_to_drop = max(xlag, ulag)
    if len(v_columns) > 0:
        vdf = df[v_columns].copy().shift(2)
        vdf.columns = 'v_' + vdf.columns
        vdf = [vdf]
        rows_to_drop = max(rows_to_drop, 2)

    x = pd.concat(xdf[:-1] + vdf + udf, axis=1).iloc[rows_to_drop:]
    y = xdf[-1].iloc[rows_to_drop:]
    return (x, y)


def df_list_to_train_test(df_list, test_index, x_columns, u_columns, xlag=None, ulag=None):

    """given a list of dataframes and a list of indices for this list saying which
    ones should be used for testing, return the training and test sets in X and y
    dataframes.

    Parameters
    ----------

    df_list : list of pandas dataframes

      list containing dataframes of time series data

    test_index : list of ints

      list of indexes into df_list saying which of them
      should be treated as the test set. All others will be treated as the
      training set.

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

    ((xtrain, ytrain), (xtest, ytest))

      xtrain, ytrain is the training set as data frames split into input (x) and output (y)

      xtest, ytest is the testing set in the same format as the training set

    """

    test_list = [df for i, df in enumerate(df_list)
                 if i in test_index]
    train_list = [df for i, df in enumerate(df_list)
                  if i not in test_index]

    xy_train = [df_to_xy(df, x_columns, u_columns, xlag=xlag, ulag=ulag)
                for df in train_list]
    xy_test = [df_to_xy(df, x_columns, u_columns, xlag=xlag, ulag=ulag)
               for df in test_list]

    return ([pd.concat(x, ignore_index=True) for x in list(zip(*xy_train))],
            [pd.concat(x, ignore_index=True) for x in list(zip(*xy_test))])


def df_to_train_test(df, experiment_column, test_experiments, x_columns, u_columns, v_columns=[], xlag=None, ulag=None):
    """given a single dataframe that has a column indicating the experiment, split into train and test

    Parameters
    ----------

    df : pandas dataframe

      data frame containing time series data

    experiment_column : str

      name of column containing experiment number

    test_experiments : list

      list of experiment numbers that should be treated as the test set. All
      others will be treated as the training set.

    x_columns : list of strings

      list of columns in the dataframe that make up the state or $x$ vector

    u_columns : list of strings

      list of columns in the dataframe that make up the control or $u$ vector

    v_columns : list of strings
      optional list of columns in the dataframe that are treated as u_columns but are lagged once so that the t-1 value is provided.

    xlag : int

      number of time steps of state $x$ to include

    ulag : int

      number of time steps of control $u$ to include

    Returns
    -------

    ((xtrain, ytrain), (xtest, ytest))

      xtrain, ytrain is the training set as data frames split into input (x) and output (y)

      xtest, ytest is the testing set in the same format as the training set



    """

    assert isinstance(experiment_column, str)
    assert isinstance(df, pd.DataFrame)
    experiments = df[experiment_column].unique()

    train_set = set(experiments) - set(test_experiments)

    xy_train = [df_to_xy(df[df[experiment_column] == exp], x_columns, u_columns, v_columns=v_columns, xlag=xlag, ulag=ulag)
                for exp in train_set]
    xy_test = [df_to_xy(df[df[experiment_column] == exp], x_columns, u_columns, v_columns=v_columns, xlag=xlag, ulag=ulag)
               for exp in test_experiments]

    return ([pd.concat(x, ignore_index=True) for x in list(zip(*xy_train))],
            [pd.concat(x, ignore_index=True) for x in list(zip(*xy_test))])


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

    xy = [df_to_xy(df[df[group_column] == exp],
                   x_columns,
                   u_columns,
                   v_columns=[],
                   xlag=xlag,
                   ulag=ulag)
          for exp in groups]

    # generate group column
    glist = np.repeat(groups, repeats=[len(x) for x, y in xy])

    X, y = [pd.concat(x, ignore_index=True) for x in list(zip(*xy))]
    return X, y, glist
