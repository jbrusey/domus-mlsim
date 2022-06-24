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

from enum import IntEnum

import numpy as np
from simple_pid import PID

from .cols import KELVIN

MAXTEMP = 999
MINTEMP = -999

DEFAULT_SETPOINT = 22 + KELVIN
PTC_P = 600
PTC_I = 0.02
PTC_D = 5
PTC_MAX = 6000
PTC_VENT_TARGET = 60 + KELVIN

COMPRESSOR_P = -1000
COMPRESSOR_I = -1
COMPRESSOR_D = -10
COMPRESSOR_MAX = 3000
COMPRESSOR_VENT_TARGET = 15 + KELVIN

MAX_RECIRC_TIME = 600 - 30
MAX_FRESH_TIME = 30

BLOWER_MULT = 17
BLOWER_ADD = 94


class SimpleHvac:

    Ut = IntEnum(
        "Ut",
        [
            "cabin_humidity",
            "cabin_temperature",
            "setpoint",
            "vent_temperature",
            "window_temperature",
        ],
        start=0,
    )
    Xt = IntEnum(
        "Xt",
        [
            "blower_level",
            "compressor_power",
            "heater_power",
            "fan_power",
            "recirc",
            "window_heating",
            "dist_defrost",
        ],
        start=0,
    )

    UT_COLUMNS = [x.name for x in Ut]
    XT_COLUMNS = [x.name for x in Xt]

    UT_MIN = np.array(
        [0, -20 + KELVIN, 15 + KELVIN, 0 + KELVIN, -20 + KELVIN], dtype=np.float32
    )
    UT_MAX = np.array(
        [1, 80 + KELVIN, 30 + KELVIN, 70 + KELVIN, 60 + KELVIN], dtype=np.float32
    )

    def __init__(self, dt=1):
        """Create a new SimpleHvac.

        All temperatures are given in kelvin.

        All humidities are given in the range 0 to 1.

        Power is in watts.

        Flap positions are in the range 0 to 1.

        Controller consists of several independent controls.

        Parameters
        ----------

        dt : integer

          Time step for each call to step method

        Description
        -----------

        SimpleHvac is a simplified controller roughly built according
        to the DOMUS WP5 specification.

        The behaviour of the controller is as follows:

        1. The blower power is set according to a hysteresis curve
        such that when the temperature is further from the setpoint,
        the power is increased. The minimum power setting is 179 and
        the maximum is 400 W.

        2. The heater is controlled using a PID controller to achieve
        a particular vent temperature when heating.

        3. The compressor is controlled using a PID controller to
        achieve a target vent temperature when cooling.

        4. Recirculation is set to operate in recirculation mode for a
        maximum of 570 seconds and then switch to fresh mode for 30
        seconds without reference to the temperatures.

        5. Window heating turns on when the dewpoint temperature of
        the cabin is under 2 degrees kelvin more than the window
        temperature.


        """
        self.dt = dt
        self.increasing_temps = np.array(
            [MINTEMP, -15, -5, -2, 5, 8, 18, MAXTEMP], dtype=np.float32
        )
        self.decreasing_temps = np.array(
            [MINTEMP, -18, -8, -5, 2, 5, 15, MAXTEMP], dtype=np.float32
        )
        # blower power is original setting (5 - 18) x 17 + 94
        self.blower_power_lu = (
            np.array([18, 18, 10, 5, 5, 10, 18, 18], dtype=np.float32) * BLOWER_MULT
            + BLOWER_ADD
        )
        self.ptc_pid = PID(
            PTC_P,
            PTC_I,
            PTC_D,
            setpoint=PTC_VENT_TARGET,
            sample_time=0,
            output_limits=(0.0, PTC_MAX),
        )
        self.compressor_pid = PID(
            COMPRESSOR_P,
            COMPRESSOR_I,
            COMPRESSOR_D,
            setpoint=COMPRESSOR_VENT_TARGET,
            output_limits=(0.0, COMPRESSOR_MAX),
        )
        self.recirc_time = 0
        self.heating_mode = False
        self.state = np.zeros((len(self.Xt)), dtype=np.float32)

    def step(self, action):
        """step takes as input an action vector (see Ut) and returns a state
        vector (see Xt)

        Parameters
        ----------

        action : array

          See Ut for the format and length of this array

        Returns
        -------

        array - see Xt for the format of this array

        """
        # clip
        action = np.clip(action, self.UT_MIN, self.UT_MAX)
        self.update_heating_mode(action)
        self.update_blower_level(action)
        self.update_pid(action)
        self.update_window_heating(action)
        self.update_recirc(action)

        return self.state

    def update_heating_mode(self, action):
        if (
            self.heating_mode
            and action[self.Ut.cabin_temperature] > action[self.Ut.setpoint] + 1
        ):
            self.heating_mode = False
        elif (
            not self.heating_mode
            and action[self.Ut.cabin_temperature] < action[self.Ut.setpoint] - 1
        ):
            self.heating_mode = True

    def update_blower_level(self, action):
        current_diff = action[self.Ut.cabin_temperature] - action[self.Ut.setpoint]

        if self.heating_mode:
            temps = self.increasing_temps
        else:
            temps = self.decreasing_temps

        level = np.interp(current_diff, temps, self.blower_power_lu)
        self.state[self.Xt.blower_level] = level

    def update_pid(self, action):

        # heater_power
        self.state[self.Xt.heater_power] = (
            self.ptc_pid(action[self.Ut.vent_temperature], dt=self.dt)
            if self.heating_mode
            else 0
        )

        # compressor_power
        self.state[self.Xt.compressor_power] = (
            self.compressor_pid(action[self.Ut.vent_temperature], dt=self.dt)
            if not self.heating_mode
            else 0
        )
        # fan power - set to same as cmp but rescaled
        self.state[self.Xt.fan_power] = (
            self.state[self.Xt.compressor_power] / 3000 * 400
        )

    def update_window_heating(self, action):
        # use simple dewpoint calculation given on wikipedia
        assert action[self.Ut.cabin_humidity] <= 1
        tdp = (
            action[self.Ut.cabin_temperature]
            - (1 - action[self.Ut.cabin_humidity]) * 20
        )
        # simple on / off used here temporarily
        self.state[self.Xt.window_heating] = int(
            action[self.Ut.window_temperature] - tdp < 2
        )
        # if window heating is needed, set defrost also
        self.state[self.Xt.dist_defrost] = self.state[self.Xt.window_heating]

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
