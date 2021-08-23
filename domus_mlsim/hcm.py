"""
DOMUS Holistic Comfort Model

@author K Sarfo Gyamfi
@author J Brusey
"""

import pickle

import numpy as np
import pkg_resources


def _load_hcm_model():
    """load pickle files needed for LDA"""
    return [
        pickle.load(
            open(pkg_resources.resource_filename(__name__, f"model/{x}.pickle"), "rb")
        )
        for x in ["LDAmdl", "Scaler", "LDAmdl_red", "Scaler_red"]
    ]


def load_hcm_model():
    """due to a change in 0.23 onward, we now need to set the number of
    features expected for the estimator as this was not done when the model was fitted."""

    n_features = [45, 45, 24, 24]
    mdl = _load_hcm_model()
    for n, m in zip(n_features, mdl):
        m.n_features_in_ = n

    return mdl


def local_eqt(ta, tr, va, pre_clo):
    """
    find local equivalent temperature for particular part of body

    @param ta - temperature (air) (deg C)
    @param tr - temperature (radiant) (deg C)
    @param va - air velocity (m/s)
    @param pre_clo - clothing insulation level

    """
    assert (
        -30 < ta < 100
    ), f"out of range air temperature ({ta}) - did you pass in kelvin?"
    assert (
        -30 < tr < 100
    ), f"out of range radiant temperature ({tr}) - did you pass in kelvin?"
    if va <= 0.1:
        return 0.5 * (ta + tr)
    else:
        return (
            0.55 * ta
            + 0.45 * tr
            + ((0.24 - 0.75 * np.sqrt(va)) / (1 + pre_clo)) * (36.5 - ta)
        )


def eqt(body_state, pre_clo, pre_out):
    """Madsen's variant of Nilsson's Equivalent Temperature

    @param body_state - 3 tuples of (air temperature, radiant
    temperature, air velocity) for each of head, torso, foot.

    temperatures are in degrees C

    air velocities are in m/s

    @param pre_clo - clothing insulation (clo)

    @param pre_out - outside temperature (degree Celsius)

    @returns ([T_eq_{head, torso, foot, overall}], [True if T_eq in bounds or False otherwise])

    """

    assert body_state.shape == (3, 3)

    eqt = [local_eqt(ta, tr, va, pre_clo) for ta, tr, va in body_state]

    # Overall
    eqt_overall = np.mean(eqt)  # (eqt_head + eqt_trunk + eqt_feet)/3
    eqt.append(eqt_overall)

    winter_bounds = [(11, 28), (13.2, 30.38), (18, 31), (18, 27.5)]
    summer_bounds = [(11, 34), (16.08, 32.16), (17, 30), (20, 30)]

    if pre_out <= 15:
        bounds = winter_bounds
    else:
        bounds = summer_bounds

    eqt_out = np.array(
        [
            bounds[bodypart][0] <= eqt[bodypart] <= bounds[bodypart][1]
            for bodypart in range(4)
        ]
    )

    return eqt, eqt_out


def binary_comfort(input_vars, input_vars_red, eqt_out, model=None):
    """
    DOMUS holistic comfort model

    """

    LDAmdl, scaler, LDAmdl_red, scaler_red = model

    # encode eqt_out as 2 if true or 1 otherwise and reshape to a column vector

    eqt_out = np.array([2 if x else 1 for x in eqt_out]).reshape(1, -1)
    eqt_out_overall = eqt_out[0, 3]
    eqt_out_head_to_feet = eqt_out[0, 0:3].reshape(1, -1)

    # Full set of features
    X = scaler.transform(input_vars)
    X = np.column_stack((X, eqt_out_head_to_feet))

    LDAout = LDAmdl.predict(X)
    score = LDAmdl.decision_function(X)

    Cov = LDAmdl.covariance_
    distances = np.zeros(
        [
            2,
        ]
    )
    for j in range(2):
        Mean = LDAmdl.means_[j]
        vec = (X - Mean).reshape([X.shape[1], 1])
        distances[j] = vec.T @ np.linalg.inv(Cov) @ vec
    mahal_distance = np.sqrt(np.min(distances))

    mean_dist = 5.737531984936752
    sigma_dist = 1.2003750458630877

    if (
        mahal_distance <= mean_dist + 2 * sigma_dist
        and mahal_distance >= mean_dist - 2 * sigma_dist
    ):
        HCMout = LDAout
    else:
        HCMout = eqt_out_overall

    # Reduced set of features
    X_red = scaler_red.transform(input_vars_red)
    X_red = np.column_stack((X_red, eqt_out_head_to_feet))

    LDAout_red = LDAmdl_red.predict(X_red)
    score_red = LDAmdl_red.decision_function(X_red)

    Cov_red = LDAmdl_red.covariance_
    distances = np.zeros(
        [
            2,
        ]
    )
    for j in range(2):
        Mean = LDAmdl_red.means_[j]
        vec = (X_red - Mean).reshape([X_red.shape[1], 1])
        distances[j] = vec.T @ np.linalg.inv(Cov_red) @ vec
    mahal_distance = np.sqrt(np.min(distances))
    # Lambda_red = 0.5391371205287423
    # prob_red = 1 - np.exp(-Lambda_red*mahal_distance)

    mean_dist_red = 3.8703734879872016
    sigma_dist_red = 1.4783964062076702

    if (mahal_distance <= mean_dist_red + 2 * sigma_dist_red) and (
        mahal_distance >= mean_dist_red - 2 * sigma_dist_red
    ):
        HCMout_red = LDAout_red
    else:
        HCMout_red = eqt_out_overall

    return (
        LDAout,
        score,
        LDAmdl.intercept_,
        LDAout_red,
        score_red,
        LDAmdl_red.intercept_,
        HCMout,
        HCMout_red,
    )


def binary_comfort_reduced(input_vars_red, eqt_out, model=None):
    """
    binary_comfort_reduced uses the LDA model to determine comfort.

    The basic model is an LDA model.

    """

    LDAmdl_red, scaler_red = model

    # encode eqt_out as 2 if true or 1 otherwise and reshape to a column vector

    eqt_out = np.array([2 if x else 1 for x in eqt_out]).reshape(1, -1)
    eqt_out_overall = eqt_out[0, 3]
    eqt_out_head_to_feet = eqt_out[0, 0:3].reshape(1, -1)

    # Reduced set of features
    X_red = np.column_stack(
        (scaler_red.transform(input_vars_red), eqt_out_head_to_feet)
    )

    LDAout_red = LDAmdl_red.predict(X_red)
    # score_red = LDAmdl_red.decision_function(X_red)

    Cov_red = LDAmdl_red.covariance_
    distances = np.zeros(
        [
            2,
        ]
    )
    for j in range(2):
        Mean = LDAmdl_red.means_[j]
        vec = (X_red - Mean).reshape([X_red.shape[1], 1])
        distances[j] = vec.T @ np.linalg.inv(Cov_red) @ vec
    mahal_distance = np.sqrt(np.min(distances))
    # Lambda_red = 0.5391371205287423
    # prob_red = 1 - np.exp(-Lambda_red*mahal_distance)

    mean_dist_red = 3.8703734879872016
    sigma_dist_red = 1.4783964062076702

    # mahalnobis distance is within 2 standard deviations of mean
    if np.abs(mahal_distance - mean_dist_red) <= 2 * sigma_dist_red:
        # use the LDA model result
        HCMout_red = LDAout_red
    else:
        # use the EqT model
        HCMout_red = eqt_out_overall

    return HCMout_red == 2


def hcm_reduced(
    model=None,
    pre_out=10,
    pre_clo=0.76,
    qa_ht=175,
    qa_wt=89,
    body_state=None,
    rh=50,
    co2ppm=400,
    lux=0,
    sound=0,
    light_blue=0,
    light_yellow=0,
    scent_OC=0,
    scent_Pepper=0,
):
    """hcm_reduced is a reduced form of the comfort model that restricts
    to parameters that are likely to be measurable or known

    @param body_state - 3 x 3 numpy array containing head, torso,
    foot: air temperature, mean radiant temperature, air velocity

    @param pre_out - outside temperature (degree Celsius)

    @param pre_clo - clothing insulation (clo)

    @param qa_ht - height of cabin occupant (cm)

    @param qa_wt - weight of cabin occupant (kg)

    @param rh - relative humidity (%)

    @param co2ppm - carbon dioxide concentration (ppm)

    @param sound - sound level (dBi)

    @param light_blue - binary variable indicating whether ambient
    light is blue (1 for Blue, 0 otherwise)

    @param light_yellow - binary variable indicating whether ambient
    light is yellow (1 for Yellow, 0 otherwise)

    @param scent_OC - binary variable indicating whether scent is
    Orange and Cinammon (1 for OC, 0 otherwise)

    @param scent_Pepper - binary variable indicating whether scent is
    Peppermint (1 for Peppermint, 0 otherwise)

    @param model - tuple containing model and scaler

    """

    # find equivalent temperature
    t_eq, eqt_out = eqt(body_state, pre_clo, pre_out)

    t_eq = np.array(t_eq)

    # take the first 3 values and reshape into a column vector
    t_eq_column = t_eq[:3].reshape(1, -1)

    input_vars_red = np.column_stack(
        (
            pre_out,
            pre_clo,
            qa_ht,
            qa_wt,
            body_state.T.reshape(1, -1),
            rh,
            co2ppm,
            lux,
            sound,
            light_blue,
            light_yellow,
            scent_OC,
            scent_Pepper,
            t_eq_column,
        )
    )

    return binary_comfort_reduced(input_vars_red, eqt_out, model)
