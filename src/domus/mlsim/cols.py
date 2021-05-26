"""
cols.py

Author
------
J. Brusey

Date
----
May 25, 2021

Column definitions for models

"""

from enum import IntEnum

DV0_UT_COLUMNS = sorted([
    # hvac controls
    't_HVACMain',
    'v_HVACMain',
    'recirc',
    'dist_defrost',
    # ambient environment
    'rh_a',
    'VehicleSpeed',
    't_a',
    # solar radiation (diffuse and direct)
    'rad1',
    'rad2',
])
DV0_XT_COLUMNS = sorted([
    # state variables needed to estimate comfort
    't_drvr1', 't_drvr2', 't_drvr3', 'v_drvr1', 'v_drvr2', 'v_drvr3',
    'm_drvr1', 'm_drvr2', 'm_drvr3',
    't_psgr1', 't_psgr2', 't_psgr3',
    'v_psgr1', 'v_psgr2', 'v_psgr3', 'm_psgr1', 'm_psgr2', 'm_psgr3',
    't_psgr21', 't_psgr22', 't_psgr23', 'v_psgr21', 'v_psgr22',
    'v_psgr23', 'm_psgr21', 'm_psgr22', 'm_psgr23', 't_psgr31',
    't_psgr32', 't_psgr33', 'v_psgr31', 'v_psgr32', 'v_psgr33',
    'm_psgr31', 'm_psgr32', 'm_psgr33',
    # humdity
    'rhc',
    # windshield temperature
    'ws',
])

DV0Ut = IntEnum('DV0Ut', DV0_UT_COLUMNS, start=0)

DV0Xt = IntEnum('DV0Xt', DV0_XT_COLUMNS, start=0)

DV1_UT_COLUMNS = sorted([
    'HvacMain',
    'vent_flow_rate',
    # hvac controls
    'recirc',
    'dist_defrost',
    # ambient environment
    'rh_a',
    'VehicleSpeed',
    't_a',
    # solar radiation (diffuse and direct)
    'rad1',
    'rad2',
    # new parameters
    'new_air_mode_Bi-Level (S.O. Middle Low)',
    'new_air_mode_Bi-Level (S.O. Side High)',
    'new_air_mode_Bi-Level (S.O. Side Low)',
    'new_air_mode_Defrost (S.O. Defrost)',
    'new_air_mode_Floor (S.O. Defrost)',
    'new_air_mode_Floor-Defrost (S.O. Defrost)',
    'new_air_mode_Panel Only (S.O. Middle High)',
    'new_air_mode_Panel Only (S.O. Middle Low)',
    'new_air_mode_Panel Only (S.O. Side High)',
    'new_air_mode_Panel Only (S.O. Side Low)',
    'radiant_panel_1',
    'radiant_panel_2',
    'radiant_panel_3',
    'radiant_panel_4',
    'seat_off',
    'seat_ventilate',
    'smart_vent_diffuse-low',
    'window_heating',
])
DV1_XT_COLUMNS = sorted([
    # state variables needed to estimate comfort
    't_drvr1', 't_drvr2', 't_drvr3', 'v_drvr1', 'v_drvr2', 'v_drvr3',
    'm_drvr1', 'm_drvr2', 'm_drvr3',
    't_psgr1', 't_psgr2', 't_psgr3',
    'v_psgr1', 'v_psgr2', 'v_psgr3', 'm_psgr1', 'm_psgr2', 'm_psgr3',
    't_psgr21', 't_psgr22', 't_psgr23', 'v_psgr21', 'v_psgr22',
    'v_psgr23', 'm_psgr21', 'm_psgr22', 'm_psgr23', 't_psgr31',
    't_psgr32', 't_psgr33', 'v_psgr31', 'v_psgr32', 'v_psgr33',
    'm_psgr31', 'm_psgr32', 'm_psgr33',
    # humdity
    'rh_c',
    # windshield temperature
    'ws',
])

DV1Ut = IntEnum('DV1Ut', DV0_UT_COLUMNS, start=0)

DV1Xt = IntEnum('DV1Xt', DV0_XT_COLUMNS, start=0)

HVAC_UT_COLUMNS = sorted([
    'blw_power',
    'cmp_power',
    'fan_power',
    'recirc',
    'ambient',
    'humidity',
    'speed',
    'solar',
    'cab_T',
    'hv_heater'
])
HVAC_XT_COLUMNS = sorted([
    'cab_RH',
    'evp_mdot',
    'vent_T',
])

HvacUt = IntEnum('HvacUt', HVAC_UT_COLUMNS, start=0)

HvacXt = IntEnum('HvacXt', HVAC_XT_COLUMNS, start=0)
