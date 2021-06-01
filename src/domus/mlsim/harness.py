"""harness.py

Author
------
J. Brusey

Date
----
May 27, 2021

Description
-----------

Connect simulators and controller together and run for a certain
number of timesteps under specific conditions.

"""


import numpy as np

from domus.control.simple_hvac import SimpleHvac
from .cols import (HvacUt,
                   HvacXt,
                   DV0Ut,
                   DV0Xt,
                   HVAC_XT_COLUMNS,
                   HVAC_UT_COLUMNS,
                   HVAC_UT_MIN,
                   HVAC_UT_MAX,
                   DV0_XT_COLUMNS,
                   DV0_UT_COLUMNS,
                   DV0_UT_MIN,
                   DV0_UT_MAX
                   )
from .mlsim import MLSim
from .util import kw_to_array


def estimate_cabin_temperature(b_x):
    """ estimate the cabin temperature based on the average front bench temperatures.
    Assumes DV0 model
    """
    assert len(b_x) == len(DV0Xt)
    return np.mean(b_x[[DV0Xt.t_drvr1,
                        DV0Xt.t_drvr2,
                        DV0Xt.t_drvr3,
                        DV0Xt.t_psgr1,
                        DV0Xt.t_psgr2,
                        DV0Xt.t_psgr3]])


def update_control_inputs(c_u, b_x, h_x, cab_t):
    """ update control vector based on cabin, hvac, and front bench temperatures.
    Assumes DV0 model.
    """
    c_u[SimpleHvac.Ut.cabin_temperature] = cab_t

    c_u[
        SimpleHvac.Ut.window_temperature
    ] = b_x[
        DV0Xt.ws
    ]

    c_u[[
        SimpleHvac.Ut.cabin_humidity,
        SimpleHvac.Ut.vent_temperature
    ]] = h_x[[
        HvacXt.cab_RH,
        HvacXt.vent_T
    ]]


def update_hvac_inputs(h_u, c_x, cab_t):
    """ update hvac input vector based on control, cabin, and front bench temperatures.
    """
    h_u[HvacUt.cab_T] = cab_t

    h_u[[
        HvacUt.blw_power,
        HvacUt.cmp_power,
        HvacUt.hv_heater,
        HvacUt.fan_power,
        HvacUt.recirc,
    ]] = c_x[[
        SimpleHvac.Xt.blower_level,
        SimpleHvac.Xt.compressor_power,
        SimpleHvac.Xt.heater_power,
        SimpleHvac.Xt.fan_power,
        SimpleHvac.Xt.recirc,
    ]]


def update_dv0_inputs(b_u, h_x, c_x):
    """ update dv0 input vector b_u based on hvac state h_x and control state c_x.
    """
    b_u[[
        DV0Ut.t_HVACMain,
        DV0Ut.v_HVACMain,
    ]] = h_x[[
        HvacXt.vent_T,
        HvacXt.evp_mdot,
    ]]

    b_u[DV0Ut.recirc] = c_x[SimpleHvac.Xt.recirc]


def make_dv0_sim(cabin_model, cabin_scaler, cabin_state):
    return MLSim(cabin_model,
                 cabin_scaler,
                 initial_state=np.vstack([cabin_state] * 2),
                 xlag=2,
                 ulag=2,
                 xlen=len(DV0_XT_COLUMNS),
                 ulen=len(DV0_UT_COLUMNS),
                 ut_min=DV0_UT_MIN,
                 ut_max=DV0_UT_MAX)


def make_hvac_sim(hvac_model, hvac_scaler, hvac_state):
    return MLSim(hvac_model,
                 hvac_scaler,
                 initial_state=np.vstack([hvac_state]),
                 xlag=1,
                 ulag=1,
                 xlen=len(HVAC_XT_COLUMNS),
                 ulen=len(HVAC_UT_COLUMNS),
                 ut_min=HVAC_UT_MIN,
                 ut_max=HVAC_UT_MAX)


def run_dv0_sim(cabin_model,
                cabin_scaler,
                hvac_model,
                hvac_scaler,
                controller,
                setpoint,
                n,
                ambient_t,
                ambient_rh,
                cabin_t,
                cabin_v,
                cabin_rh,
                solar1,
                solar2,
                car_speed):
    b_x = kw_to_array(DV0_XT_COLUMNS,
                      t_drvr1=cabin_t,
                      t_drvr2=cabin_t,
                      t_drvr3=cabin_t,
                      t_psgr1=cabin_t,
                      t_psgr2=cabin_t,
                      t_psgr3=cabin_t,
                      t_psgr21=cabin_t,
                      t_psgr22=cabin_t,
                      t_psgr23=cabin_t,
                      t_psgr31=cabin_t,
                      t_psgr32=cabin_t,
                      t_psgr33=cabin_t,
                      v_drvr1=cabin_v,
                      v_drvr2=cabin_v,
                      v_drvr3=cabin_v,
                      v_psgr1=cabin_v,
                      v_psgr2=cabin_v,
                      v_psgr3=cabin_v,
                      v_psgr21=cabin_v,
                      v_psgr22=cabin_v,
                      v_psgr23=cabin_v,
                      v_psgr31=cabin_v,
                      v_psgr32=cabin_v,
                      v_psgr33=cabin_v,
                      m_drvr1=cabin_t,
                      m_drvr2=cabin_t,
                      m_drvr3=cabin_t,
                      m_psgr1=cabin_t,
                      m_psgr2=cabin_t,
                      m_psgr3=cabin_t,
                      m_psgr21=cabin_t,
                      m_psgr22=cabin_t,
                      m_psgr23=cabin_t,
                      m_psgr31=cabin_t,
                      m_psgr32=cabin_t,
                      m_psgr33=cabin_t,
                      rhc=cabin_rh,
                      ws=cabin_t)
    h_x = kw_to_array(HVAC_XT_COLUMNS,
                      cab_RH=cabin_rh,
                      evp_mdot=cabin_v,
                      vent_T=cabin_t)

    cabin_mlsim = make_dv0_sim(cabin_model, cabin_scaler, b_x)

    hvac_mlsim = make_hvac_sim(hvac_model, hvac_scaler, h_x)

    cabin = np.zeros((n, len(b_x)))
    cabin[0] = b_x
    hvac = np.zeros((n, len(h_x)))
    hvac[0] = h_x
    ctrl = np.zeros((n, len(SimpleHvac.Xt)))
    c_u = np.zeros((len(controller.Ut)))
    c_u[controller.Ut.setpoint] = setpoint
    h_u = np.zeros((len(HvacUt)))
    h_u[[
        HvacUt.ambient,
        HvacUt.humidity,
        HvacUt.solar,
        HvacUt.speed
    ]] = [
        ambient_t,
        ambient_rh,
        solar1,
        car_speed]
    b_u = np.zeros((len(DV0Ut)))
    b_u[[
        DV0Ut.t_a,
        DV0Ut.rh_a,
        DV0Ut.rad1,
        DV0Ut.rad2,
        DV0Ut.VehicleSpeed,
    ]] = [
        ambient_t,
        ambient_rh,
        solar1,
        solar2,
        car_speed / 100 * 27.778]
    for i in range(n):
        # average temperature over front bench
        cab_t = estimate_cabin_temperature(b_x)
        update_control_inputs(c_u, b_x, h_x, cab_t)
        #    print(c_u, cab_t)
        c_x = controller.step(c_u)

        # drive HVAC
        update_hvac_inputs(h_u, c_x, cab_t)
        #    print(h_u)
        _, h_x = hvac_mlsim.step(h_u)
        #    print(h_x)
        # if h_x[HvacXt.evp_mdot] < 0:
        #     h_x[HvacXt.evp_mdot] = 0
        # drive cabin
        update_dv0_inputs(b_u, h_x, c_x)
        _, b_x = cabin_mlsim.step(b_u)
        cabin[i] = b_x
        hvac[i] = h_x
        ctrl[i] = c_x

    return cabin, hvac, ctrl
