from domus_mlsim import load_scenarios


def test_load_scenarios():
    scenarios = load_scenarios()

    assert scenarios.loc[1].ambient_t == 272.15
    assert scenarios.loc[1].ambient_rh == 0.99
    assert scenarios.loc[1].solar2 == 0

    assert scenarios.loc[28].pre_clo == 0.38
