"""simple_hvac


Author
------
J. Brusey, 21 May 2021

Description
-----------

This is a simplified controller that is roughly based on the DOMUS WP5
controller with some important differences. First, rather than produce
a temperature target as output, this controller directly drives the
PTC and compressor power. This is somewhat unrealistic but sufficient
for simulation purposes. Second, this drives the window heating
according to the relative humidity (and thus dewpoint temperature) and
the windshield temperature.

"""

import numpy as np
from simple_pid import PID
from enum import IntEnum
MAXTEMP = 999
MINTEMP = -999

KELVIN = 273
DEFAULT_SETPOINT = 22 + KELVIN
PTC_P = 600
PTC_I = 0.02
PTC_D = 5
PTC_MAX = 6000
PTC_VENT_TARGET = 60 + KELVIN

COMPRESSOR_P = -1000
COMPRESSOR_I = 0
COMPRESSOR_D = 0
COMPRESSOR_MAX = 3000
COMPRESSOR_VENT_TARGET = 15 + KELVIN

MAX_RECIRC_TIME = 600 - 30
MAX_FRESH_TIME = 30


class SimpleHvac:

    Ut = IntEnum('Ut',
                 ['cabin_humidity',
                  'cabin_temperature',
                  'setpoint',
                  'vent_temperature',
                  'window_temperature'],
                 start=0)
    Xt = IntEnum('Xt',
                 ['blower_level',
                  'compressor_power',
                  'heater_power',
                  'recirc',
                  'window_heating'],
                 start=0)

    UT_COLUMNS = [x.name for x in Ut]
    XT_COLUMNS = [x.name for x in Xt]

    UT_MIN = np.array([0,
                       -20 + KELVIN,
                       15 + KELVIN,
                       0 + KELVIN,
                       -20 + KELVIN])
    UT_MAX = np.array([1,
                       80 + KELVIN,
                       30 + KELVIN,
                       70 + KELVIN,
                       60 + KELVIN])

    def __init__(self, dt=1, setpoint=DEFAULT_SETPOINT):
        """ Create a new SimpleHvac.

        All temperatures are given in kelvin.

        Controller consists of several independent controls.

        Parameters
        ----------

        dt : integer

          Time step for each call to step method

        setpoint : float

          Target temperature in kelvin


        Blower Level
        ------------

        Blower is low when close to the target temperature and higher when further away.

        Hysteresis is used to avoid control jitter.

        Ptc Temperature
        ----------------

        Ptc temperature is altered according to a PID controller.


        Compressor Speed
        ----------------

        Compressor speed is altered according to a PID controlller.

        """
        self.dt = dt
        self.increasing_temps = np.array([MINTEMP, -15, -5, -2, 5, 8, 18, MAXTEMP])
        self.decreasing_temps = np.array([MINTEMP, -18, -8, -5, 2, 5, 15, MAXTEMP])
        # blower power is original setting (5 - 18) x 17 + 94
        self.blower_power_lu = [400, 400, 264, 179, 179, 264, 400, 400]
        self.last_cabin_temperature = 0
        self.cabin_temperature = 0
        self.vent_temperature = 0
        self.cabin_humidity = 0.5
        self.target_cabin_temperature = setpoint
        self.ptc_pid = PID(PTC_P, PTC_I, PTC_D,
                           setpoint=PTC_VENT_TARGET,
                           sample_time=0,
                           output_limits=(0.0, PTC_MAX))
        self.compressor_pid = PID(COMPRESSOR_P, COMPRESSOR_I, COMPRESSOR_D,
                                  setpoint=COMPRESSOR_VENT_TARGET,
                                  output_limits=(0.0, COMPRESSOR_MAX))
        self.recirc_time = 0
        self.heating_mode = False
        self.state = np.zeros((len(self.Xt)))

    def step(self, action):
        """step takes as input an action vector (see Ut) and returns a state
        vector (see Xt)

        """
        # clip
        action = np.clip(action, self.UT_MIN, self.UT_MAX)
        self.update_blower_level(action)
        self.update_pid(action)
        self.update_window_heating(action)
        self.update_recirc(action)

        self.last_cabin_temperature = action[self.Ut.cabin_temperature]
        return self.state

    def update_blower_level(self, action):
        """ varies between 5 and 18 """

        current_diff = action[self.Ut.cabin_temperature] - action[self.Ut.setpoint]

        if action[self.Ut.cabin_temperature] > self.last_cabin_temperature:
            temps = self.increasing_temps
        else:
            temps = self.decreasing_temps

        level = np.interp(current_diff, temps, self.blower_power_lu)
        self.state[self.Xt.blower_level] = level

    def update_pid(self, action):
        if self.heating_mode and action[self.Ut.cabin_temperature] > action[self.Ut.setpoint] - 1:
            self.heating_mode = False
        elif not self.heating_mode and action[self.Ut.cabin_temperature] < action[self.Ut.setpoint] + 1:
            self.heating_mode = True

        # heater_power
        self.state[self.Xt.heater_power] = self.ptc_pid(action[self.Ut.vent_temperature],
                                                        dt=self.dt) if self.heating_mode else 0

        # compressor_power
        self.state[self.Xt.compressor_power] = self.compressor_pid(action[self.Ut.vent_temperature],
                                                                   dt=self.dt) if not self.heating_mode else 0

    def update_window_heating(self, action):
        # use simple dewpoint calculation given on wikipedia
        assert action[self.Ut.cabin_humidity] <= 1
        tdp = action[self.Ut.cabin_temperature] - (1 - action[self.Ut.cabin_humidity]) * 20
        # simple on / off used here temporarily
        self.state[self.Xt.window_heating] = int(action[self.Ut.window_temperature] - tdp < 2)

    def update_recirc(self, action):
        if self.state[self.Xt.recirc]:
            # if currently in recirculation mode, wait until last
            # change is 570 seconds and then go to fresh for 30,
            # otherwise leave in recirc mode.
            if self.recirc_time >= MAX_RECIRC_TIME:
                self.state[self.Xt.recirc] = 0
                self.recirc_time = 0
        else:
            if self.recirc_time >= MAX_FRESH_TIME:
                self.state[self.Xt.recirc] = 1
                self.recirc_time = 0
        self.recirc_time += self.dt
