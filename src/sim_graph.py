"""

Simulate to produce timeseries graphs

Author
------
J. Brusey

Date
----
5-nov-2020

"""
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, MultiTaskLassoCV
from sklearn.metrics import mean_squared_error, r2_score
from timeseriesgroup import TimeSeriesGroupSplit

from rbfdiffm import RBFDiffMIMORegressor

from xval import configure_xt_ut
from ux_transform import unroll_by_group

from run_ml import simulate

PICKLE = 'hvac_si.pickle.gz'

LASSO_ALPHA = 0.05

MAX_EPOCHS = 2000


def load_params(infile,
                id):
    params = pd.read_csv(infile, index_col=0)
    return params[id:id + 1]

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------


def lr_model():
    return LinearRegression()


def lasso_model():
    return make_pipeline(
        PolynomialFeatures(interaction_only=True, include_bias=False),
        MultiTaskLassoCV(alphas=np.logspace(-5, 0, 30)))


def rbfdiff_model(order_opt=1,
                  p_closest=10,
                  num_iter=100,
                  xlag=2,
                  num_centres=200):
    return RBFDiffMIMORegressor(num_centres, xlag, order_opt, p_closest, num_iter)


def mlp_model_factory(hidden_nodes=None,
                      hidden_layers=None,
                      hidden_activation=None,
                      final_activation=None,
                      drop_out=None,
                      input_dim=None,
                      output_dim=None):
    def mlp_model():
        from hyper_learn import get_model
        return get_model(hidden_nodes=hidden_nodes,
                         hidden_layers=hidden_layers,
                         hidden_activation=hidden_activation,
                         final_activation=final_activation,
                         drop_out=drop_out,
                         input_dim=input_dim,
                         output_dim=output_dim)
    return mlp_model


def mlp_model(hidden_nodes=None,
              hidden_layers=None,
              hidden_activation=None,
              final_activation=None,
              drop_out=None,
              input_dim=None,
              output_dim=None):
    from tf.keras.wrappers.scikit_learn import KerasRegressor
    from keras.callbacks import EarlyStopping
    early_stop_cb = [EarlyStopping(monitor='val_loss', patience=20)]

    return KerasRegressor(build_fn=mlp_model_factory(hidden_nodes=hidden_nodes,
                                                     hidden_layers=hidden_layers,
                                                     hidden_activation=hidden_activation,
                                                     final_activation=final_activation,
                                                     drop_out=drop_out,
                                                     input_dim=input_dim,
                                                     output_dim=output_dim),
                          epochs=MAX_EPOCHS,
                          validation_split=0.2,
                          callbacks=early_stop_cb,
                          verbose=0,
                          batch_size=32)

# ------------------------------------------------------------


def learn_model(model,
                x, y, xt, yt, verbose=False):
    train_time = time.time()
    model.fit(x, y)
    train_time = time.time() - train_time

    yhat = model.predict(xt)
    if verbose:
        print(f'mse: {mean_squared_error(yt, yhat)}, R-squared: {r2_score(yt, yhat)} train time:{train_time:.2f} secs')
    return model


def graph_eval(yt, yhat, x_columns=None, exp=None, prefix='pred_'):
    yhatdf = pd.DataFrame(yhat, columns=x_columns)
    ytdf = pd.DataFrame(yt, columns=x_columns)
    yhatdf.columns = prefix + yhatdf.columns
    yth = pd.concat([ytdf, yhatdf], axis=1)

    for col in x_columns:
        _ = yth.plot(y=[col, prefix + col])
        plt.savefig(f'figures/sim_{exp}_{col}.png')
        plt.cla()

    # mse calculation
    return mean_squared_error(yhat, yt, multioutput='raw_values')


def plot_forecast_horizon(yt, yhat, title=''):
    plt_limit = 140
    MSEs = np.mean((yt - yhat)**2, axis=1)

    plt.plot(MSEs[0:plt_limit], 'bo-')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Forecast horizon')
    plt.ylabel('MSE')
    plt.show()


def simulate_with_lags(xt, yt,
                       x_len,
                       u_len,
                       model,
                       xlag,
                       ulag,
                       verbose=False):
    """Parameters
    ----------

    xt, yt : np.array

      test data with xt containing x_len * xlag + u_len * ulag
      columns while yt contains x_len columns

    x_len, u_len: scalar

      number of columns in each lag for state (x) and control (u) parts

    model : sklearn model

      model that has been fitted

    """

    n = yt.shape[0]

    # take action from columns after state columns
    action = xt[:, x_len * xlag: x_len * xlag + u_len * ulag]

    assert action.shape == (n, u_len * ulag), \
        f'{action.shape} not {n}, {u_len} x {ulag}'

    yhat = simulate(n=n,
                    initial_state=xt[0, 0:x_len * xlag],
                    action=action,
                    predictor=model.predict,
                    xlag=xlag,
                    ulag=ulag,
                    xt_len=x_len,
                    ut_len=u_len)

    return yhat


def run_cross_val(modelfn, xlag, ulag, verbose=False):

    df = pd.read_pickle(PICKLE)
    df = df.reset_index()

    scaler = MinMaxScaler()

    xc, uc = configure_xt_ut()

    sc_cols = xc + uc

    scaled_df = df[sc_cols]

    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df),
                             columns=sc_cols,
                             index=df.index)
    scaled_df = scaled_df.assign(uc=df.uc)

    X, y, groups = unroll_by_group(scaled_df,
                                   group_column='uc',
                                   x_columns=xc,
                                   u_columns=uc,
                                   xlag=xlag,
                                   ulag=ulag)

    X = X.to_numpy()
    y = y.to_numpy()

    cv = TimeSeriesGroupSplit(n_splits=10)
    split = 0
    for train, test in cv.split(X, y, groups=groups):
        split += 1
        model = modelfn()
        if verbose:
            print(f'Learning for test split {split}')
        _ = learn_model(model, X[train], y[train], X[test], y[test], verbose=verbose)

        # select out the first group's contiguous time series
        gtest = groups[test]
        xt = X[test][gtest == gtest[0]]
        yt = y[test][gtest == gtest[0]]

        yhat = simulate_with_lags(xt,
                                  yt,
                                  x_len=len(xc),
                                  u_len=len(uc),
                                  model=model,
                                  xlag=xlag,
                                  ulag=ulag,
                                  verbose=verbose)
        plot_forecast_horizon(yt, yhat, title=f'Forecast for split {split}')


def main():
    sns.set(rc={'figure.figsize': (12, 6)})
    sns.set_palette('Set2')

    import argparse
    parser = argparse.ArgumentParser(description='simulation graphs')
    parser.add_argument('--mlp',
                        action='store_true',
                        help='use MLP predictor')
    parser.add_argument('--lr',
                        action='store_true',
                        help='use LR predictor')
    parser.add_argument('--lasso',
                        action='store_true',
                        help='use LR predictor')
    parser.add_argument('--rbfdiff',
                        action='store_true',
                        help='use rbf diff net predictor')
    parser.add_argument('--xlag',
                        type=int,
                        default=2,
                        help='number of x lags')
    parser.add_argument('--ulag',
                        type=int,
                        default=1,
                        help='number of u lags')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='increase verbosity')

    args = parser.parse_args()
    if not args.mlp and not args.lr and not args.lasso and not args.rbfdiff:
        parser.print_help()
        return
    if args.mlp:
        params_df = load_params('best_param.csv', 0)

        model = mlp_model_factory(**(params_df
                                     .to_dict('records')[0]))

    elif args.lr:
        model = lr_model
    elif args.lasso:
        model = lasso_model
    elif args.rbfdiff:
        model = rbfdiff_model

    run_cross_val(model, xlag=args.xlag, ulag=args.ulag, verbose=args.verbose)


if __name__ == "__main__":
    main()
