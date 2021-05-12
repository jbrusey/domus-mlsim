import numpy as np
from simple_pid import PID
MAXTEMP = 999
MINTEMP = -999

KELVIN = 273
WARM_AIR_OUTLET_FLOOR = 60 + KELVIN
COLD_AIR_OUTLET_FLOOR = 15 + KELVIN
DEFAULT_SETPOINT = 22 + KELVIN
PTC_P = 8
PTC_I = 0.02
PTC_D = 5

COMPRESSOR_P = -1000
COMPRESSOR_I = 0
COMPRESSOR_D = 0

MAX_RECIRC_TIME = 600 - 30
MAX_FRESH_TIME = 30


class HvacController:
    def __init__(self, setpoint=DEFAULT_SETPOINT):
        """ Create a new HvacController.

        All temperatures are given in kelvin.

        Controller consists of several independent controls.

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
        self.increasing_temps = np.array([MINTEMP, -15, -5, -2, 5, 8, 18, MAXTEMP])
        self.decreasing_temps = np.array([MINTEMP, -18, -8, -5, 2, 5, 15, MAXTEMP])
        self.speed = [18, 18, 10, 5, 5, 10, 18, 18]
        self.last_cabin_temperature = 0
        self.cabin_temperature = 0
        self.cabin_humidity = 0.5
        self.target_cabin_temperature = setpoint
        self.ptc_pid = PID(PTC_P, PTC_I, PTC_D,
                           setpoint=setpoint,
                           sample_time=0,
                           output_limits=(0.0, 12.0))
        self.compressor_pid = PID(COMPRESSOR_P, COMPRESSOR_I, COMPRESSOR_D,
                                  setpoint=setpoint,
                                  output_limits=(0.0, 3000.0))
        self.heater_amps = 0.0
        self.compressor_power = 0.0
        self.window_heating = 0
        self.recirc = 1
        self.recirc_time = 0

    def setpoint(self, target):
        self.target_cabin_temperature = target
        self.ptc_pid.setpoint = target
        self.compressor_pid.setpoint = target

    def set_state(self,
                  cabin_temperature=None,
                  cabin_humidity=None,
                  window_temperature=None,
                  dt=None):
        if cabin_temperature:
            self.last_cabin_temperature = self.cabin_temperature
            self.cabin_temperature = cabin_temperature

        if cabin_humidity:
            self.cabin_humidity = cabin_humidity
        if window_temperature:
            self.window_temperature = window_temperature
        self.update_pid(dt=dt)
        self.update_window_heating()
        self.update_recirc(dt=dt)

    def blower_level(self):
        current_diff = self.cabin_temperature - self.target_cabin_temperature

        if self.cabin_temperature > self.last_cabin_temperature:
            temps = self.increasing_temps
        else:
            temps = self.decreasing_temps

        level = np.interp(current_diff, temps, self.speed)

        return level

    def update_pid(self, dt=None):
        self.heater_amps = self.ptc_pid(self.cabin_temperature, dt=dt)
        self.compressor_power = self.compressor_pid(self.cabin_temperature, dt=dt)

    def update_window_heating(self):
        # use simple dewpoint calculation given on wikipedia
        tdp = self.cabin_temperature - (1 - self.cabin_humidity) * 20
        # simple on / off used here temporarily
        self.window_heating = int(self.cabin_temperature - tdp < 2)

    def update_recirc(self, dt=None):
        self.recirc_time += dt
        if self.recirc:
            # if currently in recirculation mode, wait until last
            # change is 570 seconds and then go to fresh for 30,
            # otherwise leave in recirc mode.
            if self.recirc_time >= MAX_RECIRC_TIME:
                self.recirc = 0
                self.recirc_time = 0
        else:
            if self.recirc_time >= MAX_FRESH_TIME:
                self.recirc = 1
                self.recirc_time = 0
