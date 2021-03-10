"""

Cross validation learning

Author
------
J. Brusey

Date
----
28-oct-2020

"""

from ux_transform import df_to_train_test


def configure_xt_ut():
    """ define the state and control vectors in terms of their column names

    Returns
    -------
    xt, ut - two lists of strings

    """

    ut_columns = sorted([
        'blw_power',
        'cmp_power',
        'fan_power',
        'target_temp',
        'recirc',
        'ambient',
        'humidity',
        'speed',
        'solar',
        'cab_T',
    ])
    xt_columns = sorted([
        'cab_RH',
        'vent_T',
        'evp_air_T',
 #       'cmp_speed', #  - greatly increases mse
        'evp_mdot',
        'phigh',
        'plow',
    ])
    return (xt_columns, ut_columns)


def cross_validate_ml(df, learn_func, xlag=2, ulag=1):
    """ cross validate learning on data frame

    Parameters
    ----------

    df : data frame
      experimental data

    learn_func : (x, y), (xt, yt), exp=None
      function taking training and test data
      

    Returns
    -------

    list of results as they are produced by learn_func

    """
    experiments = df.uc.unique()
    x_columns, u_columns = configure_xt_ut()

    results = []
    for test_exp in experiments:
        (xtrain, ytrain), (xtest, ytest) = \
            df_to_train_test(df,
                             experiment_column='uc',
                             test_experiments=[test_exp],
                             x_columns=x_columns,
                             u_columns=u_columns,
                             xlag=xlag,
                             ulag=ulag)
        result = learn_func(xtrain.to_numpy(),
                            ytrain.to_numpy(),
                            xtest.to_numpy(),
                            ytest.to_numpy(),
                            exp=test_exp)
        results.append([test_exp] + list(result))

    return results


# ------------------------------------------------------------
# learning functions
# ------------------------------------------------------------




def split_by_cmp_power(df):
    """

    split dataframe according to whether the average compressor power is zero for the whole of the use case.

    Returns
    -------
    cdf, ncdf 

      where cdf is use cases where the compressor is on (cmp_power > 0)
      and ncdf is use cases where the compressor is off (cmp_power <= 0)


    """
    gdf = df.groupby('uc')

    cmp_mean = pd.DataFrame(gdf.cmp_power.aggregate(np.mean))
    cmp_mean = cmp_mean.reset_index()
    cmpuc = cmp_mean[cmp_mean.cmp_power > 0].uc
    cdf = pd.DataFrame(cmpuc).set_index('uc').assign(cmp=True)

    df1 = df.join(cdf, on='uc')

    return (df1[lambda x: ~x.cmp.isnull()].drop(columns=['cmp']),
            df1[lambda x: x.cmp.isnull()].drop(columns=['cmp']))


def skip_first(df, t):
    return df[lambda x: x.Time >= t]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='cross validation utility')
    parser.add_argument('--shape',
                        action='store_true',
                        help='provide dimensions of X and y')
    parser.add_argument('--xlag',
                        default=2,
                        type=int,
                        help='number of lags for state (x)')
    parser.add_argument('--ulag',
                        default=1,
                        type=int,
                        help='number of lags for control (u)')

    args = parser.parse_args()

    if args.shape:
        x_columns, u_columns = configure_xt_ut()
        x_len = len(x_columns) * args.xlag + len(u_columns) * args.ulag
        y_len = len(x_columns)
        print(f'X width is {x_len}, Y width is {y_len}')


if __name__ == "__main__":
    main()
