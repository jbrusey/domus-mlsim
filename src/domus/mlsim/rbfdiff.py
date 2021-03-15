""" RBF Diff Net

Author
------

Kojo Sarfo Gyamfi
James Brusey


"""

import numpy as np
from scipy.special import binom, factorial
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import sys

from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LassoCV, LinearRegression as LR

from sklearn.base import RegressorMixin, BaseEstimator

sys.setrecursionlimit(10**6)


def variableBetas(X, c, p, random_state=42):
    """
    The function returns c RBF centres and widths using K-Means clustering, with each RBF having its own unique width.


    Parameters
    ----------

    X : n x d numpy array

      input matrix of size n-by-d, where n is the number of training examples, and d is the number of features

    c : int

      the number of RBF centres

    p : int

      number of nearest neighbours to use for each centre


    Returns
    -------

    centres : output of size c-by-d

      centres determined by K-means

    betas : output of size 1-by-c

      "width" (or radius) of cluster $1 / (2 sigma^2)$

    """
    clusterObj = KMeans(n_clusters=c, random_state=random_state).fit(X)
    labels = clusterObj.labels_
    d = X.shape[1]
    betas = np.zeros((c,))
    centres = np.zeros((c, d))
    for label in range(c):
        Xk = X[labels == label, :]
        Muk = np.mean(Xk, axis=0)
        # note: Muk should be equal to clusterObj.cluster_centers_[label]
        tree = KDTree(Xk)
        dref, iref = tree.query(Muk.reshape(1, -1), k=p)
        sigma = np.mean(dref)
        betas[label] = 1 / (2 * sigma**2)
        centres[label] = Muk
    # TODO consider asserting : not np.isinf(betas).any()
    betas[np.isinf(betas)] = np.sum(betas[~np.isinf(betas)])
    return centres, betas


def rbf(X, centres, betas):
    """
    This function evaluates the radial basis function (RBF) Phi for each data sample in X.

    X: input matrix of size N-by-d, where N is the number of training examples, and d is the number of features
    centres: input array of RBF locations of size: c-by-dy, where c is the number of centres
    betas: input array of RBF widths of size: 1-by-d
    Phi: an output array of size N-by-c
    """
    N, d = X.shape
    Phi = [np.exp(-betas * np.linalg.norm(X[n] - centres, axis=1)**2) for n in range(N)]
    return np.array(Phi)


def trainRBFN(Xtrain, ytrain, c, p, normalise):
    """
    This function trains the weights for the normalised and unnormalised RBF networks

    Xtrain: training input of size n_train-by-d
    ytrain: training output of size n_train-by-1
    c: number of radial basis function (RBF) centres
    normalise: a Boolean input indicating normalised RBF network (True) or unnormalised RBF network (False)
    outputs: linear model at output layer, together with the RBF centres and widths
    """
    n, d = Xtrain.shape
    centres, betas = variableBetas(Xtrain, c, p)
    Phi = rbf(Xtrain, centres, betas)
    if normalise:
        Phi = Phi / np.sum(Phi, axis=1).reshape(-1, 1)
    linMdl = LassoCV().fit(Phi, ytrain)
    return linMdl, centres, betas


def testRBFN(Xtest, centres, betas, linMdl, normalise):
    """
    This function tests the normalised/unnormalised RBF network on new samples

    Xtest: test input of size n_test-by-d
    centres: input array of RBF locations of size: c-by-d, where c is the number of centres
    betas: input array of RBF widths of size: 1-by-d
    linMdl: linear model at output layer of RBF network
    normalise: a Boolean input indicating normalised RBF network (True) or unnormalised RBF network (False)
    outputs: one step predictions of size: n_test-by-1 - for the sequence
    """

    n, d = Xtest.shape
    Phi = rbf(Xtest, centres, betas)
    if normalise:
        Phi = Phi / np.sum(Phi, axis=1).reshape(-1, 1)
    ypred = linMdl.predict(Phi)
    return ypred


def rbfderivatives(X, centres, betas, order, Phi):
    """
    This function evaluates the radial basis function (RBF) derivatives with respect to each component of x
    up to a differential order of "order", where x is given by each row of X.

    X: input matrix of size N-by-d, where N is the number of training examples, and d is the number of features
    centres: input array of RBF locations of size: c-by-dy, where c is the number of centres
    betas: input array of RBF widths of size: 1-by-d
    Phi: input array of radial basis functions of size N-by-c
    order: input scalar representing the order of the differential equation
    diPhi: output array of RBF derivatives of size: N-by-order-by-d-by-c
    """
    N, d = X.shape
    c = centres.shape[0]
    diPhi = []
    for n in range(N):
        diPhi_n = []
        diPhi_n.append(np.tile(Phi[n], (d, 1)))
        for i in range(1, order + 1):
            leibniz_sum = 0
            for k in range(i):
                bin_coeff = binom(i - 1, k)
                if i - k - 1 == 0:
                    u = (-2 * betas.reshape(-1, 1) * (X[n] - centres)).T
                elif i - k - 1 == 1:
                    u = (-2 * betas.reshape(-1, 1) * np.ones([c, d])).T
                elif i - k - 1 >= 2:
                    u = np.zeros([d, c])
                leibniz_sum = leibniz_sum + (bin_coeff * u * diPhi_n[k])
            diPhi_n.append(leibniz_sum)
        diPhi.append(diPhi_n[1:])    # Excluding the zeroth derivative
    return np.array(diPhi)


def obj_func(x, extraArgs):
    """
    The function evaluates the loss function for training the differential RBF network weights.

    x: initial solution of weights to optimise
    extraArgs: extra arguments to compute loss function
    output: loss function value
    """
    diPhi = extraArgs[0]
    y = extraArgs[1]
    ylagged = extraArgs[2]
    n, _, c = diPhi.shape
    nlags = ylagged.shape[1]
    w_lagged = x[0:nlags].reshape(-1, 1)
    rbf_coeffs = x[nlags:nlags + c].reshape(-1, 1)
    pde_coeffs = x[nlags + c:].reshape(-1, 1)
    ypred = ((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1)) @ rbf_coeffs) + ylagged @ w_lagged
    return MSE(y, ypred)


def trainRBFDiffNet(Xtrain, ytrain, c, nlags, num_xcols, order, p, num_iter):
    """
    This function trains the differential RBF network (RBF-DiffNet)

    Xtrain: training input of size n_train-by-d
    ytrain: training output of size n_train-by-1
    c: number of radial basis function (RBF) centres
    p: number of closest neighbours to use to define centre
    nlags: number of lags or lookback window of the timeseries
    num_xcols: number of different state variables in x_columns
    order: order of the partial differential equation
    outputs: network weights together with the RBF centres and widths
    """
    n_train, d = Xtrain.shape
    centres, betas = variableBetas(Xtrain, c, p)

    lagged_indices = [count * num_xcols for count in range(nlags)]  # num_xcols is the number of different state variables in x_columns
    ylagged = Xtrain[:, lagged_indices]

    if nlags == 1:
        ylagged = ylagged.reshape(-1, 1)
    Phi = rbf(Xtrain, centres, betas)
    diPhi = rbfderivatives(Xtrain, centres, betas, order, Phi).reshape([n_train, order * d, c])

    rbfMdl = LR(fit_intercept=True).fit(Phi, ytrain)
    rbf_coeffs = rbfMdl.coef_.reshape(-1, 1)
    bias = rbfMdl.intercept_
    w_lagged = np.ones([nlags, 1]) / nlags
    # pde_coeffs = np.zeros([order * d, 1])
    pde_coeffs = (np.ones([order, d]) * (0.1**np.arange(1, order + 1) / factorial(np.arange(1, order + 1))).reshape(-1, 1)).reshape(-1, 1)
    ytrain = ytrain.reshape(-1, 1)

    ypred = ((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1)) @ rbf_coeffs) + ylagged @ w_lagged
    min_err = MSE(ytrain, ypred)
    opt_weights = [rbf_coeffs, pde_coeffs, w_lagged, bias]

    for i in range(num_iter):
        # print(i)
        # Step 1: Fix w_lagged, rbf_coeffs; solve for pde_coeffs
        pde_coeffs = np.linalg.pinv((np.sum(rbf_coeffs.reshape(1, 1, c) * diPhi, axis=2))) @ (ytrain - ylagged @ w_lagged)

        # Step 2: Fix rbf_coeffs, pde_coeffs; solve for w_lagged
        w_lagged = np.linalg.pinv(ylagged) @ (ytrain - ((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1)) @ rbf_coeffs))

        # Step 3: Fix w_lagged, pde_coeffs; solve for rbf_coeffs
        rbf_coeffs = np.linalg.pinv((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1))) @ (ytrain - ylagged @ w_lagged)

        bias = np.mean(ytrain - Phi @ rbf_coeffs)

        # Compute predictions and errors
        ypred = ((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1)) @ rbf_coeffs) + ylagged @ w_lagged
        err = MSE(ytrain, ypred)

        if err < min_err:
            min_err = err
            opt_weights = [rbf_coeffs, pde_coeffs, w_lagged, bias]

    return opt_weights, centres, betas


def testRBFDiffNet(Xtest, num_xcols, weights, centres, betas):
    """
    This function tests the differential RBF network on new samples

    Xtest: test input of size n_test-by-d
    weights: differential RBF network weights
    centres: input array of RBF locations of size: c-by-d, where c is the number of centres
    betas: input array of RBF widths of size: 1-by-d
    order: order of the partial differential equation
    outputs: one step predictions of size: n_test-by-1 - for the sequence
    """
    [rbf_coeffs, pde_coeffs, w_lagged, bias] = weights
    c = len(betas)
    n_test, d = Xtest.shape
    order = int(len(pde_coeffs) / d)
    nlags = len(w_lagged)
    lagged_indices = [count * num_xcols for count in range(nlags)]
    ylagged = Xtest[:, lagged_indices]
    Phi = rbf(Xtest, centres, betas)
    diPhi = rbfderivatives(Xtest, centres, betas, order, Phi).reshape([n_test, order * d, c])
    ypred = ((np.sum(pde_coeffs.reshape(1, len(pde_coeffs), 1) * diPhi, axis=1)) @ rbf_coeffs) + ylagged @ w_lagged
    # ypred = Phi@rbf_coeffs + bias
    return ypred


class RBFDiffRegressor(RegressorMixin, BaseEstimator):
    """

    RBFDiffRegressor

    """
    def __init__(self, c, nlags, num_xcols, order, p, num_iter):
        self.c = c
        self.nlags = nlags
        self.num_xcols = num_xcols
        self.order = order
        self.p = p
        self.num_iter = num_iter

    def fit(self, x, y):
        self.weights, self.centres, self.betas = trainRBFDiffNet(x, y,
                                                                 self.c,
                                                                 self.nlags,
                                                                 self.num_xcols,
                                                                 self.order,
                                                                 self.p,
                                                                 self.num_iter)

    def predict(self, x):
        return testRBFDiffNet(x, self.num_xcols, self.weights, self.centres, self.betas)
