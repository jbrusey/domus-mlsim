"""
test_simple_hvac

Author
------
J. Brusey

Date
----
May 6, 2021


"""

from domus_mlsim.simple_hvac import KELVIN, SimpleHvac
from domus_mlsim.util import kw_to_array

# from pytest import approx


def test_blower_level():
    control = SimpleHvac(dt=10)

    # increasing
    # cabin -18
    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 - 18 + KELVIN,
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=22 - 18 + KELVIN,
        )
    )
    assert x[SimpleHvac.Xt.blower_level] == 18 * 17 + 94

    # cabin -5
    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 - 5 + KELVIN,
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=22 - 18 + KELVIN,
        )
    )
    assert x[SimpleHvac.Xt.blower_level] == 10 * 17 + 94

    # this tests the linear interpolation
    # cabin -3.5
    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 + (-5 - 2) / 2 + KELVIN,
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=22 - 18 + KELVIN,
        )
    )
    assert x[SimpleHvac.Xt.blower_level] == (10 + 5) / 2 * 17 + 94

    # decreasing
    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 + 2 + KELVIN,
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=22 - 18 + KELVIN,
        )
    )
    assert x[SimpleHvac.Xt.blower_level] == 5 * 17 + 94


def test_vent_temperature():
    control = SimpleHvac(dt=1)

    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 - 18 + KELVIN,
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=22 - 18 + KELVIN,
        )
    )
    assert x[SimpleHvac.Xt.heater_power] == 6000.0
    assert x[SimpleHvac.Xt.compressor_power] == 0.0

    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 - 17 + KELVIN,
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=49 + KELVIN,  # <<<
        )
    )
    assert x[SimpleHvac.Xt.heater_power] == 6000.0
    assert x[SimpleHvac.Xt.compressor_power] == 0.0

    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 - 17 + KELVIN,
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=59 + KELVIN,  # <<<
        )
    )
    assert 0 < x[SimpleHvac.Xt.heater_power] < 600.0
    assert x[SimpleHvac.Xt.compressor_power] == 0.0

    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 + 5 + KELVIN,  # <<<
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=59 + KELVIN,
        )
    )
    assert x[SimpleHvac.Xt.heater_power] == 0.0
    assert x[SimpleHvac.Xt.compressor_power] == 3000.0

    assert not control.heating_mode
    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 - 1 + KELVIN,  # <<< limit of not cold enough to switch
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=22 + KELVIN,
        )
    )
    assert not control.heating_mode
    assert x[SimpleHvac.Xt.heater_power] == 0
    assert x[SimpleHvac.Xt.compressor_power] > 0

    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 - 1.1 + KELVIN,  # <<< now cold enough
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=22 + KELVIN,
        )
    )
    assert control.heating_mode
    assert x[SimpleHvac.Xt.heater_power] > 0
    assert x[SimpleHvac.Xt.compressor_power] == 0

    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 + 1 + KELVIN,  # <<< at limit
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=22 + KELVIN,
        )
    )
    assert control.heating_mode
    assert x[SimpleHvac.Xt.heater_power] > 0
    assert x[SimpleHvac.Xt.compressor_power] == 0

    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 + 1.1 + KELVIN,  # <<< above limit
            setpoint=22 + KELVIN,
            window_temperature=22 - 18 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=22 + KELVIN,
        )
    )
    assert not control.heating_mode
    assert x[SimpleHvac.Xt.heater_power] == 0
    assert x[SimpleHvac.Xt.compressor_power] > 0


def test_demist():
    control = SimpleHvac()

    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 + KELVIN,
            setpoint=22 + KELVIN,
            window_temperature=22 + KELVIN,
            cabin_humidity=0.5,
            vent_temperature=59 + KELVIN,
        )
    )
    assert x[SimpleHvac.Xt.window_heating] == 0.0

    x = control.step(
        kw_to_array(
            SimpleHvac.UT_COLUMNS,
            cabin_temperature=22 + KELVIN,
            setpoint=22 + KELVIN,
            window_temperature=2 + KELVIN,  # <<<
            cabin_humidity=0.9,  # <<<
            vent_temperature=59 + KELVIN,
        )
    )
    assert x[SimpleHvac.Xt.window_heating] == 1.0


def test_recirc():
    control = SimpleHvac(dt=10)

    clock = 0
    for i in range(120):
        x = control.step(
            kw_to_array(
                SimpleHvac.UT_COLUMNS,
                cabin_temperature=22 + KELVIN,
                setpoint=22 + KELVIN,
                window_temperature=2 + KELVIN,  # <<<
                cabin_humidity=0.9,  # <<<
                vent_temperature=59 + KELVIN,
            )
        )
        #        print(f'{clock}: {x[SimpleHvac.Xt.recirc]}')
        assert x[SimpleHvac.Xt.recirc] == int((clock % 600) >= 30)
        clock += 10
