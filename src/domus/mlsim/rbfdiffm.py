""" RBFDiffMIMORegressor

Author
------
Kojo Sarfo Gyamfi
James Brusey

Revised version of base MISO style regressor that supports MIMO (multiple columns in output)

"""

import numpy as np
from scipy.special import factorial
import sys

from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression as LR

from sklearn.base import RegressorMixin, BaseEstimator
from .rbfdiff import rbf, variableBetas, rbfderivatives

sys.setrecursionlimit(10**6)


def trainRBFDiffNetMISO(Xtrain, ytrain, ylagged, Phi, diPhi, order, d, num_iter):
    """
    This function trains the differential RBF network (RBF-DiffNet)

    Xtrain: training input of size n_train-by-d
    ytrain: training output of size n_train-by-1
    c: number of radial basis function (RBF) centres
    nlags: number of lags or lookback window of the timeseries
    order: order of the partial differential equation
    outputs: network weights together with the RBF centres and widths
    """

    c = Phi.shape[1]
    nlags = ylagged.shape[1]
    rbfMdl = LR(fit_intercept=True).fit(Phi, ytrain)
    rbf_coeffs = rbfMdl.coef_.reshape(-1, 1)
    # TODO check - variable is assigned but never used
    # bias = rbfMdl.intercept_
    w_lagged = np.ones([nlags, 1]) / nlags
    # pde_coeffs = np.zeros([order * d, 1])
    pde_coeffs = (np.ones([order, d]) * (0.1**np.arange(1, order + 1) / factorial(np.arange(1, order + 1))).reshape(-1, 1)).reshape(-1, 1)
    ytrain = ytrain.reshape(-1, 1)

    ypred = ((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1)) @ rbf_coeffs) + ylagged @ w_lagged
    min_err = MSE(ytrain, ypred)
    opt_weights = [rbf_coeffs, pde_coeffs, w_lagged]

    for i in range(num_iter):
        # print(i)
        # Step 1: Fix w_lagged, rbf_coeffs; solve for pde_coeffs
        pde_coeffs = np.linalg.pinv((np.sum(rbf_coeffs.reshape(1, 1, c) * diPhi, axis=2))) @ (ytrain - ylagged @ w_lagged)

        # Step 2: Fix rbf_coeffs, pde_coeffs; solve for w_lagged
        w_lagged = np.linalg.pinv(ylagged) @ (ytrain - ((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1)) @ rbf_coeffs))

        # Step 3: Fix w_lagged, pde_coeffs; solve for rbf_coeffs
        rbf_coeffs = np.linalg.pinv((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1))) @ (ytrain - ylagged @ w_lagged)

        # TODO Check - var is assigned but never used
        # bias = np.mean(ytrain - Phi @ rbf_coeffs)

        # Compute predictions and errors
        ypred = ((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1)) @ rbf_coeffs) + ylagged @ w_lagged
        err = MSE(ytrain, ypred)

        if err < min_err:
            min_err = err
            opt_weights = [rbf_coeffs, pde_coeffs, w_lagged]

    return opt_weights


def trainRBFDiffNetMIMO(X_train, Y_train, c, nlags, order, p, num_iter, verbose=False):
    n_train, d = X_train.shape
    centres, betas = variableBetas(X_train, c, p)
    Phi = rbf(X_train, centres, betas)
    diPhi = rbfderivatives(X_train, centres, betas, order, Phi).reshape([n_train, order * d, c])

    WEIGHTS = {}
    for i in range(Y_train.shape[1]):
        if verbose:
            print('training output', i)
        lagged_indices = [i + count * Y_train.shape[1] for count in range(nlags)]

        y_lagged = X_train[:, lagged_indices]
        if nlags == 1:
            y_lagged = y_lagged.reshape(-1, 1)
        y_train = Y_train[:, i].reshape(-1, 1)

        opt_weights = trainRBFDiffNetMISO(X_train, y_train, y_lagged, Phi, diPhi, order, d, num_iter)
        WEIGHTS[i] = opt_weights

    return WEIGHTS, centres, betas


def testRBFDiffNetMIMO(X_test, nlags, WEIGHTS, centres, betas):
    YPRED = []
    num_out = len(WEIGHTS)
    for i in range(num_out):
        weights = WEIGHTS[i]
        lagged_indices = [i + count * num_out for count in range(nlags)]
        ylagged = X_test[:, lagged_indices]
        ypred = testRBFDiffNetMISO(X_test, ylagged, weights, centres, betas).reshape(-1, 1)
        YPRED.append(ypred)
    YPREDICT = np.column_stack(tuple(YPRED))
    return YPREDICT


def testRBFDiffNetMISO(Xtest, ylagged, weights, centres, betas):
    """
    This function tests the differential RBF network on new samples

    Xtest: test input of size n_test-by-d
    weights: differential RBF network weights
    centres: input array of RBF locations of size: c-by-d, where c is the number of centres
    betas: input array of RBF widths of size: 1-by-d
    order: order of the partial differential equation
    outputs: one step predictions of size: n_test-by-1 - for the sequence
    """
    [rbf_coeffs, pde_coeffs, w_lagged] = weights
    c = len(betas)
    n_test, d = Xtest.shape
    order = int(len(pde_coeffs) / d)
    # TODO check - variable is assigned but never used
    # nlags = len(w_lagged)
    Phi = rbf(Xtest, centres, betas)
    diPhi = rbfderivatives(Xtest, centres, betas, order, Phi).reshape([n_test, order * d, c])
    ypred = ((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1)) @ rbf_coeffs) + ylagged @ w_lagged
    return ypred


class RBFDiffMIMORegressor(RegressorMixin, BaseEstimator):
    """

    RBFDiffMIMORegressor

    """
    def __init__(self, c, nlags, order, p, num_iter):
        self.c = c
        self.nlags = nlags
        self.order = order
        self.p = p
        self.num_iter = num_iter

    def fit(self, x, y):
        self.weights, self.centres, self.betas = trainRBFDiffNetMIMO(x, y,
                                                                     self.c,
                                                                     self.nlags,
                                                                     self.order,
                                                                     self.p,
                                                                     self.num_iter)

    def predict(self, x):
        return testRBFDiffNetMIMO(x,
                                  self.nlags,
                                  self.weights,
                                  self.centres,
                                  self.betas)
