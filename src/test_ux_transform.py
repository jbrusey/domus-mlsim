""" test for ux_transform

Author
------
J. Brusey

Date
----
27/10/2020

"""

import pandas as pd
import numpy as np

from ux_transform import (df_to_xy,
                          df_list_to_train_test,
                          df_to_train_test,
                          unroll_by_group)


def test_df_to_xy():

    df = pd.DataFrame({'a': [101, 102, 103, 104, 105],
                       'b': [201, 204, 209, 216, 225],
                       'c': [301, 308, 327, 381, 543],
                       'd': [405, 406, 407, 408, 409]})

    x, y = df_to_xy(df,
                    ['a', 'b'],
                    ['c', 'd'], xlag=2, ulag=1)
    print(df)
    print(x)
    print(y)

    assert (x.columns.values == ['x1_a',
                                 'x1_b',
                                 'x2_a',
                                 'x2_b',
                                 'u1_c',
                                 'u1_d']).all()

    assert (y.columns.values == ['x3_a',
                                 'x3_b']).all()

    assert len(x) == 3
    assert len(y) == 3

    assert (x.x1_a == np.array([101, 102, 103])).all()
    assert (y.x3_a == np.array([103, 104, 105])).all()


def test_df_list_to_train_test():

    df_list = [pd.DataFrame({'a': [101, 102, 103, 104, 105],
                             'b': [201, 204, 209, 216, 225],
                             'c': [301, 308, 327, 381, 543],
                             'd': [405, 406, 407, 408, 409]}),

               pd.DataFrame({'a': [501, 502, 503, 504, 505],
                             'b': [201, 204, 209, 216, 225],
                             'c': [301, 308, 327, 381, 543],
                             'd': [405, 406, 407, 408, 409]}),

               pd.DataFrame({'a': [601, 602, 603, 604, 605],
                             'b': [201, 204, 209, 216, 225],
                             'c': [301, 308, 327, 381, 543],
                             'd': [405, 406, 407, 408, 409]}),
               ]

    (xtrain, ytrain), (xtest, ytest) = df_list_to_train_test(df_list,
                                                             [0],
                                                             ['a', 'b'],
                                                             ['c'],
                                                             xlag=2,
                                                             ulag=1
                                                             )
    assert len(xtrain) == 6
    assert len(xtest) == 3

    # print(xtrain)
    # print(ytrain)
    # print(xtest)
    # print(ytest)

    assert (xtrain.x1_a == np.array([501, 502, 503, 601, 602, 603])).all()
    assert (ytrain.x3_a == np.array([503, 504, 505, 603, 604, 605])).all()
    assert (xtest.x1_a == np.array([101, 102, 103])).all()
    assert (ytest.x3_a == np.array([103, 104, 105])).all()

    (xtrain, ytrain), (xtest, ytest) = df_list_to_train_test(df_list,
                                                             [1, 2],
                                                             ['a', 'b'],
                                                             ['c'],
                                                             xlag=2,
                                                             ulag=1
                                                             )
    assert (xtest.x1_a == np.array([501, 502, 503, 601, 602, 603])).all()
    assert (ytest.x3_a == np.array([503, 504, 505, 603, 604, 605])).all()
    assert (xtrain.x1_a == np.array([101, 102, 103])).all()
    assert (ytrain.x3_a == np.array([103, 104, 105])).all()


def test_df_to_train_test():

    df_list = [pd.DataFrame({'a': [101, 102, 103, 104, 105],
                             'b': [201, 204, 209, 216, 225],
                             'c': [301, 308, 327, 381, 543],
                             'd': [405, 406, 407, 408, 409]}),

               pd.DataFrame({'a': [501, 502, 503, 504, 505],
                             'b': [201, 204, 209, 216, 225],
                             'c': [301, 308, 327, 381, 543],
                             'd': [405, 406, 407, 408, 409]}),

               pd.DataFrame({'a': [601, 602, 603, 604, 605],
                             'b': [201, 204, 209, 216, 225],
                             'c': [301, 308, 327, 381, 543],
                             'd': [405, 406, 407, 408, 409]}),
               ]

    mdf = pd.concat([df.assign(kkk=i + 100)
                     for i, df in enumerate(df_list)], ignore_index=True)

    (xtrain, ytrain), (xtest, ytest) = df_to_train_test(mdf,
                                                        experiment_column='kkk',
                                                        test_experiments=[100],
                                                        x_columns=['a', 'b'],
                                                        u_columns=['c'],
                                                        xlag=2,
                                                        ulag=1)
    assert len(xtrain) == 6
    assert len(xtest) == 3

    # print(xtrain)
    # print(ytrain)
    # print(xtest)
    # print(ytest)

    assert (xtrain.x1_a == np.array([501, 502, 503, 601, 602, 603])).all()
    assert (ytrain.x3_a == np.array([503, 504, 505, 603, 604, 605])).all()
    assert (xtest.x1_a == np.array([101, 102, 103])).all()
    assert (ytest.x3_a == np.array([103, 104, 105])).all()

    (xtrain, ytrain), (xtest, ytest) = df_to_train_test(mdf,
                                                        experiment_column='kkk',
                                                        test_experiments=[101, 102],
                                                        x_columns=['a', 'b'],
                                                        u_columns=['c'],
                                                        xlag=2,
                                                        ulag=1)
    assert (xtest.x1_a == np.array([501, 502, 503, 601, 602, 603])).all()
    assert (ytest.x3_a == np.array([503, 504, 505, 603, 604, 605])).all()
    assert (xtrain.x1_a == np.array([101, 102, 103])).all()
    assert (ytrain.x3_a == np.array([103, 104, 105])).all()

    # test case where v_columns is not provided
    (x, y), (xt, yt) = df_to_train_test(mdf,
                                        experiment_column='kkk',
                                        test_experiments=[100],
                                        x_columns=['a'],
                                        u_columns=['c'],
                                        xlag=1,
                                        ulag=1)
    assert (x.x1_a == np.array([501, 502, 503, 504, 601, 602, 603, 604])).all()
    assert (x.u1_c == np.array([308, 327, 381, 543, 308, 327, 381, 543])).all()

    # test case where v_columns is provided
    (x, y), (xt, yt) = df_to_train_test(mdf,
                                        experiment_column='kkk',
                                        test_experiments=[100],
                                        x_columns=['a'],
                                        u_columns=['c'],
                                        v_columns=['b'],
                                        xlag=1,
                                        ulag=1)

    assert (x.v_b == np.array([201, 204, 209, 201, 204, 209])).all()
    assert (x.x1_a == np.array([502, 503, 504, 602, 603, 604])).all()
    assert (x.u1_c == np.array([327, 381, 543, 327, 381, 543])).all()

    assert (x.columns.values == ['x1_a', 'v_b', 'u1_c']).all()

    # check that ordering is x1, x2, ... u1, u2, ...
    (x, y), (xt, yt) = df_to_train_test(mdf,
                                        experiment_column='kkk',
                                        test_experiments=[100],
                                        x_columns=['a'],
                                        u_columns=['c'],
                                        v_columns=[],
                                        xlag=2,
                                        ulag=2)
    assert (x.columns.values == ['x1_a', 'x2_a', 'u1_c', 'u2_c']).all()


def test_unroll_by_group():

    df = pd.DataFrame({'a': [101, 102, 103, 104, 105],
                       'b': [201, 204, 209, 216, 225],
                       'c': [301, 308, 327, 381, 543],
                       'd': [405, 406, 407, 408, 409],
                       'g': [1,     1,   1,   2,   2]})

    x, y, group = unroll_by_group(df,
                                  group_column='g',
                                  x_columns=['a', 'b'],
                                  u_columns=['c', 'd'], xlag=1, ulag=1)

    assert (group == [1, 1, 2]).all()

    assert (x.columns.values == ['x1_a',
                                 'x1_b',
                                 'u1_c',
                                 'u1_d']).all()

    assert (y.columns.values == ['x2_a',
                                 'x2_b']).all()

    assert len(x) == 3
    assert len(y) == 3

    assert (x.x1_a == np.array([101, 102, 104])).all()
    assert (y.x2_a == np.array([102, 103, 105])).all()
