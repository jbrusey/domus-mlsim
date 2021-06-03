"""
test_harness

Author
------
J. Brusey

Date
----
May 27, 2021


"""

from domus.mlsim.harness import (estimate_cabin_temperature_dv0,
                                 estimate_cabin_temperature_dv1,
                                 update_control_inputs_dv0,
                                 update_control_inputs_dv1,
                                 update_hvac_inputs,
                                 update_dv0_inputs,
                                 update_dv1_inputs,
                                 make_dv0_sim,
                                 make_dv1_sim,
                                 make_hvac_sim,
                                 run_dv0_sim,
                                 run_dv1_sim,
                                 DV0Xt,
                                 DV0Ut,
                                 DV0_XT_COLUMNS,
                                 DV1Xt,
                                 DV1Ut,
                                 DV1_XT_COLUMNS,
                                 HvacXt,
                                 HvacUt,
                                 HVAC_XT_COLUMNS,
                                 )
from domus.control.simple_hvac import SimpleHvac
from domus.mlsim.cols import KELVIN
import numpy as np
from numpy.testing import assert_array_equal
import joblib


def test_estimate_cabin_temperature_dv0():

    b_x = np.zeros((len(DV0Xt)))

    assert estimate_cabin_temperature_dv0(b_x) == 0

    b_x[[
        DV0Xt.t_drvr1,
        DV0Xt.t_drvr2,
        DV0Xt.t_drvr3,
        DV0Xt.t_psgr1,
        DV0Xt.t_psgr2,
        DV0Xt.t_psgr3]] = [[
            1, 2, 3, 4, 5, 6]]
    assert estimate_cabin_temperature_dv0(b_x) == 3.5


def test_estimate_cabin_temperature_dv1():

    b_x = np.zeros((len(DV1Xt)))

    assert estimate_cabin_temperature_dv1(b_x) == 0

    b_x[[
        DV1Xt.t_drvr1,
        DV1Xt.t_drvr2,
        DV1Xt.t_drvr3,
        DV1Xt.t_psgr1,
        DV1Xt.t_psgr2,
        DV1Xt.t_psgr3]] = [[
            1, 2, 3, 4, 5, 6]]
    assert estimate_cabin_temperature_dv1(b_x) == 3.5


def test_update_control_inputs_dv0():
    b_x = np.zeros((len(DV0Xt)))

    b_x[DV0Xt.ws] = 283

    h_x = np.zeros((len(HvacXt)))
    h_x[[
        HvacXt.cab_RH,
        HvacXt.vent_T
    ]] = [0.1, 285]

    c_u = np.zeros((len(SimpleHvac.Ut)))
    update_control_inputs_dv0(c_u, b_x, h_x, 286)
    assert_array_equal(c_u, [0.1, 286, 0, 285, 283])


def test_update_control_inputs_dv1():
    b_x = np.zeros((len(DV1Xt)))

    b_x[DV1Xt.ws] = 283

    h_x = np.zeros((len(HvacXt)))
    h_x[[
        HvacXt.cab_RH,
        HvacXt.vent_T
    ]] = [0.1, 285]

    c_u = np.zeros((len(SimpleHvac.Ut)))
    update_control_inputs_dv1(c_u, b_x, h_x, 286)
    assert_array_equal(c_u, [0.1, 286, 0, 285, 283])


def test_update_hvac_inputs():
    c_x = np.zeros((len(SimpleHvac.Xt)))
    h_u = np.zeros((len(HvacUt)))

    c_x[[
        SimpleHvac.Xt.blower_level,
        SimpleHvac.Xt.compressor_power,
        SimpleHvac.Xt.heater_power,
        SimpleHvac.Xt.fan_power,
        SimpleHvac.Xt.recirc,
    ]] = [400, 3000, 6000, 200, 0.5]
    update_hvac_inputs(h_u, c_x, 286)

    expect = np.zeros((len(HvacUt)))
    expect[[
        HvacUt.cab_T,
        HvacUt.blw_power,
        HvacUt.cmp_power,
        HvacUt.hv_heater,
        HvacUt.fan_power,
        HvacUt.recirc,
    ]] = [286, 400, 3000, 6000, 200, 0.5]

    assert_array_equal(expect, h_u)


def test_update_dv0_inputs():
    c_x = np.zeros((len(SimpleHvac.Xt)))
    h_x = np.zeros((len(HvacXt)))
    b_u = np.zeros((len(DV0Ut)))

    c_x[[
        SimpleHvac.Xt.recirc,
    ]] = [0.5]
    h_x[[
        HvacXt.vent_T,
        HvacXt.evp_mdot,
    ]] = [273, 180]

    update_dv0_inputs(b_u, h_x, c_x)

    expect = np.zeros((len(DV0Ut)))
    expect[[
        DV0Ut.t_HVACMain,
        DV0Ut.v_HVACMain,
        DV0Ut.recirc,
    ]] = [273, 180, 0.5]

    assert_array_equal(expect, b_u)


def test_update_dv1_inputs():
    c_x = np.zeros((len(SimpleHvac.Xt)))
    h_x = np.zeros((len(HvacXt)))
    b_u = np.zeros((len(DV1Ut)))

    c_x[[
        SimpleHvac.Xt.recirc,
        SimpleHvac.Xt.blower_level,
        SimpleHvac.Xt.window_heating
    ]] = [0.5, 264, 1.0]
    h_x[[
        HvacXt.vent_T,
        HvacXt.evp_mdot,
    ]] = [273, 180]

    update_dv1_inputs(b_u, h_x, c_x)

    expect = np.zeros((len(DV1Ut)))
    expect[[
        DV1Ut.HvacMain,
        DV1Ut.vent_flow_rate,
        DV1Ut.recirc,
        DV1Ut.window_heating,
        DV1Ut.smart_vent_diffuse_low,
        DV1Ut.new_air_mode_Floor_SO_Defrost,
        DV1Ut.seat_off,
    ]] = [273, 3, 0.5, 1.0, 1, 1, 1]

    assert_array_equal(expect, b_u)


def test_make_dv0_sim():
    ROOT = '../'
    DV0_MODEL = ROOT + 'model/3d_lr.joblib'

    dv0_scaler, dv0_model = joblib.load(DV0_MODEL)

    dv0_state = np.zeros((len(DV0_XT_COLUMNS)))

    dv0_sim = make_dv0_sim(dv0_model, dv0_scaler, dv0_state)

    assert dv0_sim is not None


def test_make_dv1_sim():
    ROOT = '../'
    DV1_MODEL = ROOT + 'model/dv1_lr.joblib'

    dv1_scaler, dv1_model = joblib.load(DV1_MODEL)

    dv1_state = np.zeros((len(DV1_XT_COLUMNS)))

    dv1_sim = make_dv1_sim(dv1_model, dv1_scaler, dv1_state)

    assert dv1_sim is not None


def test_make_hvac_sim():
    ROOT = '../'
    HVAC_MODEL = ROOT + 'model/hvac_lr.joblib'

    hvac_scaler, hvac_model = joblib.load(HVAC_MODEL)

    hvac_state = np.zeros((len(HVAC_XT_COLUMNS)))

    hvac_sim = make_hvac_sim(hvac_model, hvac_scaler, hvac_state)

    assert hvac_sim is not None


def test_run_dv0_sim():
    ROOT = '../'
    DV0_MODEL = ROOT + 'model/3d_lr.joblib'
    HVAC_MODEL = ROOT + 'model/hvac_lr.joblib'
    dv0_scaler, dv0_model = joblib.load(DV0_MODEL)
    hvac_scaler, hvac_model = joblib.load(HVAC_MODEL)

    cabin, hvac, ctrl = run_dv0_sim(dv0_model,
                                    dv0_scaler,
                                    hvac_model,
                                    hvac_scaler,
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
                                    car_speed=100)

    # temperature should increase
    assert cabin[50, DV0Xt.t_drvr1] > cabin[0, DV0Xt.t_drvr1]


def test_run_dv1_sim():
    ROOT = '../'
    DV1_MODEL = ROOT + 'model/dv1_lr.joblib'
    HVAC_MODEL = ROOT + 'model/hvac_lr.joblib'
    dv1_scaler, dv1_model = joblib.load(DV1_MODEL)
    hvac_scaler, hvac_model = joblib.load(HVAC_MODEL)

    cabin, hvac, ctrl = run_dv1_sim(dv1_model,
                                    dv1_scaler,
                                    hvac_model,
                                    hvac_scaler,
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
                                    car_speed=100)

    # temperature should increase
    assert cabin[50, DV1Xt.t_drvr1] > cabin[0, DV1Xt.t_drvr1]
