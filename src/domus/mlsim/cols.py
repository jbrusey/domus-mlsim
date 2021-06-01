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
from .util import kw_to_array

KELVIN = 273.15

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
    'new_air_mode_Bi_Level_SO_Side_Low',
    'new_air_mode_Panel_Only_SO_Side_Low',
    'new_air_mode_Panel_Only_SO_Side_High',
    'new_air_mode_Panel_Only_SO_Middle_High',
    'new_air_mode_Defrost_SO_Defrost',
    'new_air_mode_Bi_Level_SO_Side_High',
    'new_air_mode_Floor_SO_Defrost',
    'new_air_mode_Floor_Defrost_SO_Defrost',
    'new_air_mode_Panel_Only_SO_Middle_Low',
    'new_air_mode_Bi_Level_SO_Middle_Low',
    'radiant_panel_1',
    'radiant_panel_2',
    'radiant_panel_3',
    'radiant_panel_4',
    'seat_off',
    'seat_ventilate',
    'smart_vent_diffuse_low',
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

# ------------------------------------------------------------

# minimums
DV0_UT_MIN = kw_to_array(DV0_UT_COLUMNS,
                         t_HVACMain=KELVIN + 5,
                         v_HVACMain=0,
                         recirc=0,
                         dist_defrost=0,
                         rh_a=0,
                         VehicleSpeed=0,
                         t_a=KELVIN - 20,
                         rad1=0,
                         rad2=0)
# maximums
DV0_UT_MAX = kw_to_array(DV0_UT_COLUMNS,
                         t_HVACMain=KELVIN + 60,
                         v_HVACMain=300,
                         recirc=1,
                         dist_defrost=1,
                         rh_a=1,
                         VehicleSpeed=28,
                         t_a=KELVIN + 50,
                         rad1=170,
                         rad2=120)

DV1_UT_MIN = kw_to_array(DV1_UT_COLUMNS,
                         HvacMain=KELVIN + 5,
                         vent_flow_rate=0,
                         recirc=0,
                         dist_defrost=0,
                         rh_a=0,
                         VehicleSpeed=0,
                         t_a=KELVIN - 20,
                         rad1=0,
                         rad2=0,
                         new_air_mode_Bi_Level_SO_Side_Low=0,
                         new_air_mode_Panel_Only_SO_Side_Low=0,
                         new_air_mode_Panel_Only_SO_Side_High=0,
                         new_air_mode_Panel_Only_SO_Middle_High=0,
                         new_air_mode_Defrost_SO_Defrost=0,
                         new_air_mode_Bi_Level_SO_Side_High=0,
                         new_air_mode_Floor_SO_Defrost=0,
                         new_air_mode_Floor_Defrost_SO_Defrost=0,
                         new_air_mode_Panel_Only_SO_Middle_Low=0,
                         new_air_mode_Bi_Level_SO_Middle_Low=0,
                         radiant_panel_1=0,
                         radiant_panel_2=0,
                         radiant_panel_3=0,
                         radiant_panel_4=0,
                         seat_off=0,
                         seat_ventilate=0,
                         smart_vent_diffuse_low=0,
                         window_heating=0,
                         )

DV1_UT_MAX = kw_to_array(DV1_UT_COLUMNS,
                         HvacMain=KELVIN + 60,
                         vent_flow_rate=300,
                         recirc=1,
                         dist_defrost=1,
                         rh_a=1,
                         VehicleSpeed=28,
                         t_a=1,
                         rad1=170,
                         rad2=120,
                         new_air_mode_Bi_Level_SO_Side_Low=1,
                         new_air_mode_Panel_Only_SO_Side_Low=1,
                         new_air_mode_Panel_Only_SO_Side_High=1,
                         new_air_mode_Panel_Only_SO_Middle_High=1,
                         new_air_mode_Defrost_SO_Defrost=1,
                         new_air_mode_Bi_Level_SO_Side_High=1,
                         new_air_mode_Floor_SO_Defrost=1,
                         new_air_mode_Floor_Defrost_SO_Defrost=1,
                         new_air_mode_Panel_Only_SO_Middle_Low=1,
                         new_air_mode_Bi_Level_SO_Middle_Low=1,
                         radiant_panel_1=1,
                         radiant_panel_2=1,
                         radiant_panel_3=1,
                         radiant_panel_4=1,
                         seat_off=1,
                         seat_ventilate=1,
                         smart_vent_diffuse_low=1,
                         window_heating=1,
                         )

HVAC_UT_MIN = kw_to_array(HVAC_UT_COLUMNS,
                          blw_power=179,
                          cmp_power=0,
                          fan_power=0,
                          recirc=0,
                          ambient=KELVIN - 20,
                          humidity=0,
                          speed=0,
                          solar=0,
                          cab_T=KELVIN - 20,
                          hv_heater=0)
HVAC_UT_MAX = kw_to_array(HVAC_UT_COLUMNS,
                          blw_power=400,
                          cmp_power=3200,
                          fan_power=420,
                          recirc=1,
                          ambient=KELVIN - 50,
                          humidity=1,
                          speed=100,
                          solar=1000,
                          cab_T=KELVIN + 80,
                          hv_heater=6000)
