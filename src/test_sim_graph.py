import numpy as np

from pytest import approx

import pandas as pd
from sklearn.metrics import mean_squared_error

from sim_graph import (lr_model,
                       learn_model,
                       simulate_with_lags)

from ux_transform import unroll_by_group


def test_lr_model():

    x = np.array([[1, 2, 3, 4]]).T

    y = np.array([[0.5, 0.6, 0.7, 0.8]]).T

    model = lr_model()
    _ = model.fit(x, y)

    yhat = model.predict(np.array([[5]]))

    assert yhat[0, 0] == approx(0.9)


def test_simulate_with_lags():

    # y = a exp{-k t} + c
    # dy/dt = k(c - y)

    N = 100
    t = np.arange(0, N + 1)
    k = 0.1
    a = 10
    c = 5
    yt = a * np.exp(-k * t) + c

    x = yt[:-1]
    y = yt[1:]

    df = pd.DataFrame({'t': t[:-1], 'y': x})
    df = df.assign(u=0)
    df = df.assign(y2=y)
    pct40 = ((N * 4) // 10)
    df = df.assign(uc=[0] * pct40 + [1] * (N - pct40))
    df = df.dropna()

    X, y, groups = unroll_by_group(df,
                                   group_column='uc',
                                   x_columns=['y'],
                                   u_columns=['u'],
                                   xlag=2,
                                   ulag=2)

    X = X.to_numpy()
    y = y.to_numpy()

    xx, yy = X[groups == 0], y[groups == 0]

    xt, yt = X[groups == 1], y[groups == 1]

    model = lr_model()
    _ = learn_model(model, xx, yy, xt, yt)

    yhat = simulate_with_lags(xt, yt,
                              model=model,
                              x_len=1,
                              u_len=1,
                              xlag=2,
                              ulag=2)

    for i in range(yhat.shape[0]):
        assert approx(yhat[i]) == yt[i]
    assert mean_squared_error(yhat, yt) < 1e-4
