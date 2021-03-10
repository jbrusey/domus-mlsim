
from partial_scaler import PartialScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from numpy.testing import assert_array_almost_equal


def test_partial_scaler():

    df = pd.DataFrame({'a': [101, 102, 103, 104, 105],
                       'b': [201, 204, 209, 216, 225],
                       'c': [301, 308, 327, 381, 543],
                       'd': [405, 406, 407, 408, 409]})

    X = df.to_numpy()
    s = MinMaxScaler()
    s.fit(X)

    orig = X[2:3, :]
    
    t = s.transform(orig)

    assert_array_almost_equal(s.inverse_transform(t), orig)

    orig = X[2:4, :2]

    px = PartialScaler(s, 0, 2, n_features=4)
    pu = PartialScaler(s, 2, 4, n_features=4)

    t = px.transform(orig)
    assert_array_almost_equal(px.inverse_transform(t), orig)

    t = pu.transform(orig)
    assert_array_almost_equal(pu.inverse_transform(t), orig)
