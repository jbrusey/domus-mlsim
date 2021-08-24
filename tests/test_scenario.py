from domus_mlsim import load_scenarios


def test_load_scenarios():
    scenarios = load_scenarios()

    assert scenarios.loc[1].ambient_t == 273.15
    assert scenarios.loc[1].ambient_rh == 1.0
    assert scenarios.loc[1].solar2 == 0

    assert scenarios.loc[28].pre_clo == 0.38
    assert scenarios.loc[1].precond == 30
    assert scenarios.loc[1].time == 15
