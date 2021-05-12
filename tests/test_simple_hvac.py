"""
test_simple_hvac

Author
------
J. Brusey

Date
----
May 6, 2021


"""

from domus.control.simple_hvac import HvacController, KELVIN
# from pytest import approx


def test_blower_level():
    control = HvacController()

    # increasing
    control.setpoint(22 + KELVIN)
    control.set_state(cabin_temperature=22 - 18 + KELVIN,   # -18
                      dt=10
                      )
    assert control.blower_level() == 18  # for inc or dec
    control.set_state(cabin_temperature=22 - 5 + KELVIN, dt=10)    # -5
    assert control.blower_level() == 10

    # this tests the linear interpolation
    control.set_state(cabin_temperature=22 + (-5 + -2) / 2 + KELVIN, dt=10)
    assert control.blower_level() == (10 + 5) / 2

    # decreasing
    control.set_state(cabin_temperature=22 - 5 + KELVIN, dt=10)
    assert control.blower_level() == 5


def test_vent_temperature():
    control = HvacController()

    control.setpoint(22 + KELVIN)

    control.set_state(cabin_temperature=22 - 18 + KELVIN, dt=10)   # -18 (cold)
    assert control.heater_amps == 12.0  # maximum
    assert control.compressor_power == 0.0   # cooling off

    control.set_state(cabin_temperature=22 + KELVIN, dt=10)   # neutral
    assert control.heater_amps == 0   # approx(0.02 * 10 * 18)  # integral based on previous error
    assert control.compressor_power == 0.0   # cooling off

    control.set_state(cabin_temperature=22 + 5 + KELVIN, dt=10)   # hot
    assert control.heater_amps == 0  # no heating
    assert control.compressor_power == 3000.0   # cooling on max


def test_demist():
    control = HvacController()

    control.set_state(cabin_humidity=0.5, window_temperature=22, dt=10)
    assert control.window_heating == 0   # no mist so no heating

    control.set_state(cabin_humidity=0.9, window_temperature=2, dt=10)
    assert control.window_heating == 1   # at dewpoint temperature


def test_recirc():
    control = HvacController()

    clock = 0
    for i in range(120):
        control.set_state(cabin_temperature=22 + KELVIN, dt=10)
        clock += 10
        print(clock, control.recirc)
        assert control.recirc == int((clock % 600) < 600 - 30)
