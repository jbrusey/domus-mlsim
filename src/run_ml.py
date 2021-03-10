"""

run_ml.py

use learnt simulation model to simulate specific scenario that
includes control inputs



J. Brusey
April 2020

"""


import numpy as np


class BoundsException(Exception):
    def __init__(self, s):
        super(BoundsException, self).__init__(s)


def lr_predictor(reg):
    """ create a closure that predicts using linear regression
    """
    def predict(x):
        return reg.predict(x)

    return predict


def simulate(n=1,
             initial_state=None,
             action=None,
             predictor=None,
             xt_len=None,
             ut_len=None,
             xlag=None,
             ulag=None):
    """simulate n steps given an initial_state and a set of n actions.

    Parameters
    ----------

    n : int

      number of time steps to simulate for

    initial_state : np.array

      start state of shape xlag x xt_len

    action : np.array of shape n x ut_len * ulag

      list of n actions where each action is a set of ulag vectors of length ut_len each

    predictor : function (vector) -> vector

      function taking vector containing xlag previous states and ulag
      previous control actions (including current one) that returns
      next state.

      Typically this function should be created using a closure or lambda.

    xt_len : int

      length of state vector

    ut_len : int

      length of control vector

    Returns
    -------

      np.array of dimension n x xt_len

    Note: For n=1, this takes xlag previous states and ulag previous
    actions and gives the next state.

    """
    assert action.shape == (n, ut_len * ulag), \
        f'wrong action shape {action.shape}, should be ({n},{ut_len} * {ulag})'

    state_stack = initial_state.reshape(xlag, -1)
    assert state_stack.shape == (xlag, xt_len)

    # create an empty array of n rows and |s| columns
    data = np.ndarray((n, xt_len))
    for i in range(n):
        assert state_stack.shape == (xlag, xt_len), \
            f'state stack wrong shape {state_stack} != {xlag},{xt_len}'
        x = np.hstack((state_stack.reshape(1, -1),
                       action[i].reshape(1, -1)))
        assert x.shape == (1, xlag * xt_len + ulag * ut_len),\
            f'x is wrong shape {x.shape} != (1, {xlag} * {xt_len} + {ulag} * {ut_len})'
        s = predictor(x)
        state_stack = np.append(state_stack[1:],
                                s.reshape(1, -1),
                                axis=0)

        data[i] = s
    return data
