"""
test_harness

Author
------
J. Brusey

Date
----
May 27, 2021


"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_max_ulp
from sklearn.base import BaseEstimator

from domus_mlsim import (
    DV0_XT_COLUMNS,
    DV1_XT_COLUMNS,
    HVAC_XT_COLUMNS,
    KELVIN,
    DV0Ut,
    DV0Xt,
    DV1Ut,
    DV1Xt,
    HvacUt,
    HvacXt,
    MLSim,
    SimpleHvac,
    estimate_cabin_temperature_dv0,
    estimate_cabin_temperature_dv1,
    load_dv0,
    load_dv1,
    load_hvac,
    make_dv0_sim,
    make_dv1_sim,
    make_hvac_sim,
    run_dv0_sim,
    run_dv1_sim,
    update_control_inputs_dv0,
    update_control_inputs_dv1,
    update_dv0_inputs,
    update_dv1_inputs,
    update_hvac_inputs,
)


def test_estimate_cabin_temperature_dv0():

    b_x = np.zeros((len(DV0Xt)))

    assert estimate_cabin_temperature_dv0(b_x) == 0

    b_x[
        [
            DV0Xt.t_drvr1,
            DV0Xt.t_drvr2,
            DV0Xt.t_drvr3,
            DV0Xt.t_psgr1,
            DV0Xt.t_psgr2,
            DV0Xt.t_psgr3,
            DV0Xt.m_drvr1,
            DV0Xt.m_drvr2,
            DV0Xt.m_drvr3,
            DV0Xt.m_psgr1,
            DV0Xt.m_psgr2,
            DV0Xt.m_psgr3,
        ]
    ] = [[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]]
    assert estimate_cabin_temperature_dv0(b_x) == 3.5


def test_estimate_cabin_temperature_dv1():

    b_x = np.zeros((len(DV1Xt)))

    assert estimate_cabin_temperature_dv1(b_x) == 0

    b_x[
        [
            DV1Xt.t_drvr1,
            DV1Xt.t_drvr2,
            DV1Xt.t_drvr3,
            DV1Xt.t_psgr1,
            DV1Xt.t_psgr2,
            DV1Xt.t_psgr3,
            DV0Xt.m_drvr1,
            DV0Xt.m_drvr2,
            DV0Xt.m_drvr3,
            DV0Xt.m_psgr1,
            DV0Xt.m_psgr2,
            DV0Xt.m_psgr3,
        ]
    ] = [[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]]
    assert estimate_cabin_temperature_dv1(b_x) == 3.5


def test_update_control_inputs_dv0():
    b_x = np.zeros((len(DV0Xt)))

    b_x[DV0Xt.ws] = 283

    h_x = np.zeros((len(HvacXt)))
    h_x[[HvacXt.cab_RH, HvacXt.vent_T]] = [0.1, 285]

    c_u = np.zeros((len(SimpleHvac.Ut)))
    update_control_inputs_dv0(c_u, b_x, h_x, 286)
    assert_array_equal(c_u, [0.1, 286, 0, 285, 283])


def test_update_control_inputs_dv1():
    b_x = np.zeros((len(DV1Xt)))

    b_x[DV1Xt.ws] = 283

    h_x = np.zeros((len(HvacXt)))
    h_x[[HvacXt.cab_RH, HvacXt.vent_T]] = [0.1, 285]

    c_u = np.zeros((len(SimpleHvac.Ut)))
    update_control_inputs_dv1(c_u, b_x, h_x, 286)
    assert_array_equal(c_u, [0.1, 286, 0, 285, 283])


def test_update_hvac_inputs():
    c_x = np.zeros((len(SimpleHvac.Xt)), dtype=np.float32)
    h_u = np.zeros((len(HvacUt)))

    c_x[
        [
            SimpleHvac.Xt.blower_level,
            SimpleHvac.Xt.compressor_power,
            SimpleHvac.Xt.heater_power,
            SimpleHvac.Xt.fan_power,
            SimpleHvac.Xt.recirc,
            SimpleHvac.Xt.dist_defrost,
        ]
    ] = [400, 3000, 6000, 200, 0.5, 0.6]
    update_hvac_inputs(h_u, c_x, 286)

    expect = np.zeros((len(HvacUt)))
    expect[
        [
            HvacUt.cab_T,
            HvacUt.blw_power,
            HvacUt.cmp_power,
            HvacUt.hv_heater,
            HvacUt.fan_power,
            HvacUt.recirc,
        ]
    ] = [286, 400, 3000, 6000, 200, 0.5]

    assert_array_equal(expect, h_u)


def test_update_dv0_inputs():
    c_x = np.zeros((len(SimpleHvac.Xt)), dtype=np.float32)
    h_x = np.zeros((len(HvacXt)))
    b_u = np.zeros((len(DV0Ut)))

    c_x[
        [
            SimpleHvac.Xt.recirc,
            SimpleHvac.Xt.dist_defrost,
        ]
    ] = [0.5, 0.6]
    h_x[
        [
            HvacXt.vent_T,
            HvacXt.evp_mdot,
        ]
    ] = [273, 180]

    update_dv0_inputs(b_u, h_x, c_x)

    expect = np.zeros((len(DV0Ut)))
    expect[
        [
            DV0Ut.t_HVACMain,
            DV0Ut.v_HVACMain,
            DV0Ut.recirc,
            DV0Ut.dist_defrost,
        ]
    ] = [273, 180, 0.5, 0.6]

    assert_array_max_ulp(expect, b_u, dtype=np.float32)


def test_update_dv1_inputs():
    c_x = np.zeros((len(SimpleHvac.Xt)), dtype=np.float32)
    h_x = np.zeros((len(HvacXt)))
    b_u = np.zeros((len(DV1Ut)))

    c_x[
        [
            SimpleHvac.Xt.recirc,
            SimpleHvac.Xt.blower_level,
            SimpleHvac.Xt.window_heating,
            SimpleHvac.Xt.dist_defrost,
        ]
    ] = [0.5, 264, 1.0, 0.6]
    h_x[
        [
            HvacXt.vent_T,
            HvacXt.evp_mdot,
        ]
    ] = [273, 180]

    update_dv1_inputs(b_u, h_x, c_x)

    expect = np.zeros((len(DV1Ut)))
    expect[
        [
            DV1Ut.HvacMain,
            DV1Ut.vent_flow_rate,
            DV1Ut.recirc,
            DV1Ut.dist_defrost,
            DV1Ut.window_heating,
            DV1Ut.new_air_mode_Floor_SO_Defrost,
            DV1Ut.seat_off,
        ]
    ] = [273, 3, 0.5, 0.6, 1.0, 1, 1]

    assert_array_max_ulp(expect, b_u, dtype=np.float32)


@pytest.mark.parametrize(
    "func",
    [
        load_dv0,
        load_dv1,
        load_hvac,
    ],
)
def test_load(func):
    scaler_and_model = func()

    assert (
        scaler_and_model is not None
        and len(scaler_and_model) == 2
        and isinstance(scaler_and_model[0], BaseEstimator)
    )


@pytest.mark.parametrize(
    "state_len,load_func,make_func",
    [
        (len(DV0_XT_COLUMNS), load_dv0, make_dv0_sim),
        (len(DV1_XT_COLUMNS), load_dv1, make_dv1_sim),
        (len(HVAC_XT_COLUMNS), load_hvac, make_hvac_sim),
    ],
)
def test_make_sim(state_len, load_func, make_func):

    scaler_and_model = load_func()
    state = np.zeros((state_len))

    sim = make_func(scaler_and_model, state)
    assert sim is not None and isinstance(sim, MLSim)


def test_run_dv0_sim():
    cabin, hvac, ctrl = run_dv0_sim(
        load_dv0(),
        load_hvac(),
        SimpleHvac(),
        setpoint=KELVIN + 22,
        n=100,
        ambient_t=KELVIN + 1,
        ambient_rh=0.99,
        cabin_t=KELVIN + 1,
        cabin_v=0,
        cabin_rh=0.99,
        solar1=100,
        solar2=50,
        car_speed=100,
    )
    assert ctrl.dtype == np.float32

    # temperature should increase
    assert cabin[50, DV0Xt.t_drvr1] > cabin[0, DV0Xt.t_drvr1]


def test_run_dv1_sim():
    cabin, hvac, ctrl = run_dv1_sim(
        load_dv1(),
        load_hvac(),
        SimpleHvac(),
        setpoint=KELVIN + 22,
        n=100,
        ambient_t=KELVIN + 1,
        ambient_rh=0.99,
        cabin_t=KELVIN + 1,
        cabin_v=0,
        cabin_rh=0.99,
        solar1=100,
        solar2=50,
        car_speed=100,
    )
    assert ctrl.dtype == np.float32

    # temperature should increase
    assert cabin[50, DV1Xt.t_drvr1] > cabin[0, DV1Xt.t_drvr1]


def test_run_dv1_sim_log():
    cabin, hvac, ctrl, b_u_log, h_u_log, c_u_log = run_dv1_sim(
        load_dv1(),
        load_hvac(),
        SimpleHvac(),
        setpoint=KELVIN + 22,
        n=100,
        ambient_t=KELVIN + 1,
        ambient_rh=0.99,
        cabin_t=KELVIN + 1,
        cabin_v=0,
        cabin_rh=0.99,
        solar1=100,
        solar2=50,
        car_speed=100,
        log_inputs=True,
    )

    # temperature should increase
    assert cabin[50, DV1Xt.t_drvr1] > cabin[0, DV1Xt.t_drvr1]
