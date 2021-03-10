from mlsim import MLSim
from ux_transform import unroll_by_group
from sim_graph import lr_model, learn_model
import numpy as np
import pandas as pd
from pytest import approx
from sklearn.preprocessing import MinMaxScaler


def test_mlsim():
    # learn a model first
    N = 100
    t = np.arange(0, N)
    k = 0.1
    a = 10
    c = 5
    yt = a * np.exp(-k * t) + c

    df = pd.DataFrame({'t': t, 'y': yt})
    df = df.assign(u=0)
    pct40 = ((N * 4) // 10)

    xc = ['y']
    uc = ['u']

    sc_cols = xc + uc
    scaled_df = df[sc_cols]

    scaler = MinMaxScaler()

    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df),
                             columns=sc_cols,
                             index=df.index)

    scaled_df = scaled_df.assign(uc=[0] * pct40 + [1] * (N - pct40))
    scaled_df = scaled_df.dropna()
    xlag = 2
    X, y, groups = unroll_by_group(scaled_df,
                                   group_column='uc',
                                   x_columns=xc,
                                   u_columns=uc,
                                   xlag=xlag,
                                   ulag=2)

    X = X.to_numpy()
    y = y.to_numpy()

    xx, yy = X[groups == 0], y[groups == 0]

    xt, yt = X[groups == 1], y[groups == 1]

    model = lr_model()
    _ = learn_model(model, xx, yy, xt, yt)

    initial_state = df.loc[0:xlag - 1, xc].to_numpy()

    sim = MLSim(model,
                scaler=scaler,
                initial_state=initial_state,
                xlag=2,
                ulag=2,
                xlen=1,
                ulen=1)

    s1 = sim.step([0])

    assert approx(df.loc[xlag, xc].to_numpy()) == s1[0]

    s2 = sim.step([0])

    assert approx(df.loc[xlag + 1, xc].to_numpy()) == s2[0]
