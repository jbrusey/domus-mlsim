import numpy as np
from pytest import approx

from domus_mlsim import binary_comfort, eqt, load_hcm_model, hcm_reduced


def test_eqt():

    # eqt for <0.1 air velocity is just average of ta, tr

    # assume summer

    teq, bounds = eqt(
        np.array([[10, 20, 0.05], [11, 21, 0.05], [12, 22, 0.05]]), 0.7, 20
    )
    assert teq == [15.0, 16.0, 17.0, 16.0]

    assert bounds[0] and not bounds[1] and bounds[2] and not bounds[3]

    # check that the value changes when we increase air velocity >= 0.1

    teq, bounds = eqt(
        np.array([[10, 20, 0.25], [11, 21, 0.25], [12, 22, 0.25]]), 0.7, 20
    )
    assert teq[0] < 15
    assert teq[1] < 16
    assert teq[2] < 17
    assert teq[3] == np.mean(teq[:3])


def test_binary_comfort_eg1():

    pre_out = 9
    pre_t = 21
    pre_clo = 0.76
    qa_age = 25
    qa_ht = 175
    qa_wt = 73
    q2_2 = 0
    thist_d0 = 9
    tsa_q3_1 = 5
    tsa_q3_2 = 0
    tsa_q3_3 = -2
    tsa_q3_4 = 0
    tsa_q3_5 = -1
    tsa_q3_6 = -1
    tsa_q3_7 = 1
    q4_8 = 2
    q4_18 = -1
    ta_hd = 22
    ta_tr = 21.5
    ta_ft = 20
    tr_hd = 22.8
    tr_tr = 22
    tr_ft = 20.9
    va_hd = 0.1
    va_tr = 0.1
    va_ft = 0.1
    rh = 26
    co2ppm = 1400
    lux = 5.06
    sound = 61.25
    qc_1 = -8
    qc_2 = -10
    qc_3 = -6
    qc_4 = -8
    qc_5 = -10
    qc_6 = 10
    qa_gender_m = 1
    q2_1_yes = 0
    light_blue = 0
    light_yellow = 0
    scent_OC = 0
    scent_Pepper = 0

    body_state = np.array(
        [[ta_hd, tr_hd, va_hd], [ta_tr, tr_tr, va_tr], [ta_ft, tr_ft, va_ft]]
    )

    t_eq, eqt_out = eqt(body_state, pre_clo, pre_out)

    t_eq = np.array(t_eq)

    # take the first 3 values and reshape into a column vector
    t_eq_column = t_eq[:3].reshape(1, -1)

    # eqt_head, eqt_trunk, eqt_feet, eqt_out_head, eqt_out_trunk, eqt_out_feet, eqt_out_overall = eqt(ta_hd, ta_tr, ta_ft, tr_hd, tr_tr, tr_ft, va_hd, va_tr, va_ft, pre_clo, pre_out)

    input_vars = np.column_stack(
        (
            pre_out,
            pre_t,
            pre_clo,
            qa_age,
            qa_ht,
            qa_wt,
            q2_2,
            thist_d0,
            tsa_q3_1,
            tsa_q3_2,
            tsa_q3_3,
            tsa_q3_4,
            tsa_q3_5,
            tsa_q3_6,
            tsa_q3_7,
            q4_8,
            q4_18,
            ta_hd,
            ta_tr,
            ta_ft,
            tr_hd,
            tr_tr,
            tr_ft,
            va_hd,
            va_tr,
            va_ft,
            rh,
            co2ppm,
            lux,
            sound,
            qc_1,
            qc_2,
            qc_3,
            qc_4,
            qc_5,
            qc_6,
            qa_gender_m,
            q2_1_yes,
            light_blue,
            light_yellow,
            scent_OC,
            scent_Pepper,
            t_eq_column,
        )
    )

    input_vars_red = np.column_stack(
        (
            pre_out,
            pre_clo,
            qa_ht,
            qa_wt,
            ta_hd,
            ta_tr,
            ta_ft,
            tr_hd,
            tr_tr,
            tr_ft,
            va_hd,
            va_tr,
            va_ft,
            rh,
            co2ppm,
            lux,
            sound,
            light_blue,
            light_yellow,
            scent_OC,
            scent_Pepper,
            t_eq_column,
        )
    )

    (
        comfort_indicator,
        score,
        threshold,
        comfort_indicator_red,
        score_red,
        threshold_red,
        HCMout,
        HCMout_red,
    ) = binary_comfort(input_vars, input_vars_red, eqt_out, model=load_hcm_model())

    assert HCMout == 2, "occupant should be comfortable for this example"

    assert score[0] == approx(1.7733586531269188)

    assert threshold[0] == approx(-3.234199373163648)

    assert HCMout_red == 2

    assert score_red[0] == approx(0.5449404047755122)

    assert threshold_red[0] == approx(-2.5008015493225457)


def test_binary_comfort_eg2():
    pre_out = 12
    pre_t = 21
    pre_clo = 0.76
    qa_age = 48
    qa_ht = 165
    qa_wt = 57
    q2_2 = 5
    thist_d0 = 13
    tsa_q3_1 = 7
    tsa_q3_2 = -1
    tsa_q3_3 = 2
    tsa_q3_4 = -1
    tsa_q3_5 = 1
    tsa_q3_6 = 2
    tsa_q3_7 = -1
    q4_8 = -1
    q4_18 = -2
    ta_hd = 27.5
    ta_tr = 26.7
    ta_ft = 27.7
    tr_hd = 27.6
    tr_tr = 27.2
    tr_ft = 30.3
    va_hd = 0.3
    va_tr = 0.45
    va_ft = 0.17
    rh = 38
    co2ppm = 860
    lux = 5.06
    sound = 66.25
    qc_1 = -8
    qc_2 = -10
    qc_3 = -10
    qc_4 = -10
    qc_5 = -10
    qc_6 = 0
    qa_gender_m = 0
    q2_1_yes = 0
    light_blue = 0
    light_yellow = 0
    scent_OC = 0
    scent_Pepper = 0

    body_state = np.array(
        [[ta_hd, tr_hd, va_hd], [ta_tr, tr_tr, va_tr], [ta_ft, tr_ft, va_ft]]
    )

    t_eq, eqt_out = eqt(body_state, pre_clo, pre_out)

    t_eq = np.array(t_eq)

    # take the first 3 values and reshape into a column vector
    t_eq_column = t_eq[:3].reshape(1, -1)

    # eqt_head, eqt_trunk, eqt_feet, eqt_out_head, eqt_out_trunk, eqt_out_feet, eqt_out_overall = eqt(ta_hd, ta_tr, ta_ft, tr_hd, tr_tr, tr_ft, va_hd, va_tr, va_ft, pre_clo, pre_out)

    input_vars = np.column_stack(
        (
            pre_out,
            pre_t,
            pre_clo,
            qa_age,
            qa_ht,
            qa_wt,
            q2_2,
            thist_d0,
            tsa_q3_1,
            tsa_q3_2,
            tsa_q3_3,
            tsa_q3_4,
            tsa_q3_5,
            tsa_q3_6,
            tsa_q3_7,
            q4_8,
            q4_18,
            ta_hd,
            ta_tr,
            ta_ft,
            tr_hd,
            tr_tr,
            tr_ft,
            va_hd,
            va_tr,
            va_ft,
            rh,
            co2ppm,
            lux,
            sound,
            qc_1,
            qc_2,
            qc_3,
            qc_4,
            qc_5,
            qc_6,
            qa_gender_m,
            q2_1_yes,
            light_blue,
            light_yellow,
            scent_OC,
            scent_Pepper,
            t_eq_column,
        )
    )

    input_vars_red = np.column_stack(
        (
            pre_out,
            pre_clo,
            qa_ht,
            qa_wt,
            ta_hd,
            ta_tr,
            ta_ft,
            tr_hd,
            tr_tr,
            tr_ft,
            va_hd,
            va_tr,
            va_ft,
            rh,
            co2ppm,
            lux,
            sound,
            light_blue,
            light_yellow,
            scent_OC,
            scent_Pepper,
            t_eq_column,
        )
    )

    (
        comfort_indicator,
        score,
        threshold,
        comfort_indicator_red,
        score_red,
        threshold_red,
        HCMout,
        HCMout_red,
    ) = binary_comfort(input_vars, input_vars_red, eqt_out, model=load_hcm_model())

    assert HCMout == 1, "occupant should be uncomfortable for this example"

    assert score[0] == approx(-0.9369385133610963)

    assert threshold[0] == approx(-3.234199373163648)

    assert HCMout_red == 1

    assert score_red[0] == approx(-0.6477776016370358)

    assert threshold_red[0] == approx(-2.5008015493225457)


def test_binary_comfort_eg3():
    pre_out = 6
    pre_t = 20
    pre_clo = 0.76
    qa_age = 26
    qa_ht = 178
    qa_wt = 85
    q2_2 = 5
    thist_d0 = 6
    tsa_q3_1 = 2
    tsa_q3_2 = 3
    tsa_q3_3 = -1
    tsa_q3_4 = 3
    tsa_q3_5 = -2
    tsa_q3_6 = -2
    tsa_q3_7 = 2
    q4_8 = 1
    q4_18 = -1
    ta_hd = 20.1
    ta_tr = 20.1
    ta_ft = 20.1
    tr_hd = 20.1
    tr_tr = 20.1
    tr_ft = 20.1
    va_hd = 0.09
    va_tr = 0.07
    va_ft = 0.08
    rh = 41.5
    co2ppm = 45
    lux = 75
    sound = 64
    qc_1 = -9
    qc_2 = -9
    qc_3 = -5
    qc_4 = -8
    qc_5 = -8
    qc_6 = -10
    qa_gender_m = 1
    q2_1_yes = 0
    light_blue = 0
    light_yellow = 0
    scent_OC = 0
    scent_Pepper = 1

    body_state = np.array(
        [[ta_hd, tr_hd, va_hd], [ta_tr, tr_tr, va_tr], [ta_ft, tr_ft, va_ft]]
    )

    t_eq, eqt_out = eqt(body_state, pre_clo, pre_out)

    t_eq = np.array(t_eq)

    # take the first 3 values and reshape into a column vector
    t_eq_column = t_eq[:3].reshape(1, -1)

    # eqt_head, eqt_trunk, eqt_feet, eqt_out_head, eqt_out_trunk, eqt_out_feet, eqt_out_overall = eqt(ta_hd, ta_tr, ta_ft, tr_hd, tr_tr, tr_ft, va_hd, va_tr, va_ft, pre_clo, pre_out)

    input_vars = np.column_stack(
        (
            pre_out,
            pre_t,
            pre_clo,
            qa_age,
            qa_ht,
            qa_wt,
            q2_2,
            thist_d0,
            tsa_q3_1,
            tsa_q3_2,
            tsa_q3_3,
            tsa_q3_4,
            tsa_q3_5,
            tsa_q3_6,
            tsa_q3_7,
            q4_8,
            q4_18,
            ta_hd,
            ta_tr,
            ta_ft,
            tr_hd,
            tr_tr,
            tr_ft,
            va_hd,
            va_tr,
            va_ft,
            rh,
            co2ppm,
            lux,
            sound,
            qc_1,
            qc_2,
            qc_3,
            qc_4,
            qc_5,
            qc_6,
            qa_gender_m,
            q2_1_yes,
            light_blue,
            light_yellow,
            scent_OC,
            scent_Pepper,
            t_eq_column,
        )
    )

    input_vars_red = np.column_stack(
        (
            pre_out,
            pre_clo,
            qa_ht,
            qa_wt,
            ta_hd,
            ta_tr,
            ta_ft,
            tr_hd,
            tr_tr,
            tr_ft,
            va_hd,
            va_tr,
            va_ft,
            rh,
            co2ppm,
            lux,
            sound,
            light_blue,
            light_yellow,
            scent_OC,
            scent_Pepper,
            t_eq_column,
        )
    )

    (
        comfort_indicator,
        score,
        threshold,
        comfort_indicator_red,
        score_red,
        threshold_red,
        HCMout,
        HCMout_red,
    ) = binary_comfort(input_vars, input_vars_red, eqt_out, model=load_hcm_model())

    assert HCMout == 1, "occupant should be uncomfortable for this example"

    assert score[0] == approx(-0.09327184132086908)

    assert threshold[0] == approx(-3.234199373163648)

    assert HCMout_red == 1

    assert score_red[0] == approx(-0.7259478747861583)

    assert threshold_red[0] == approx(-2.5008015493225457)


def test_hcm_reduced1():

    pre_out = 9
    pre_clo = 0.76
    qa_ht = 175
    qa_wt = 73
    ta_hd = 22
    ta_tr = 21.5
    ta_ft = 20
    tr_hd = 22.8
    tr_tr = 22
    tr_ft = 20.9
    va_hd = 0.1
    va_tr = 0.1
    va_ft = 0.1
    rh = 26
    co2ppm = 1400
    lux = 5.06
    sound = 61.25
    light_blue = 0
    light_yellow = 0
    scent_OC = 0
    scent_Pepper = 0

    body_state = np.array(
        [[ta_hd, tr_hd, va_hd], [ta_tr, tr_tr, va_tr], [ta_ft, tr_ft, va_ft]]
    )

    _, _, ldamdl, scale = load_hcm_model()

    assert hcm_reduced(
        model=(ldamdl, scale),
        pre_out=pre_out,
        pre_clo=pre_clo,
        qa_ht=qa_ht,
        qa_wt=qa_wt,
        body_state=body_state,
        rh=rh,
        co2ppm=co2ppm,
        lux=lux,
        sound=sound,
        light_blue=light_blue,
        light_yellow=light_yellow,
        scent_OC=scent_OC,
        scent_Pepper=scent_Pepper,
    )


def test_hcm_reduced2():
    pre_out = 12
    pre_clo = 0.76
    qa_ht = 165
    qa_wt = 57
    ta_hd = 27.5
    ta_tr = 26.7
    ta_ft = 27.7
    tr_hd = 27.6
    tr_tr = 27.2
    tr_ft = 30.3
    va_hd = 0.3
    va_tr = 0.45
    va_ft = 0.17
    rh = 38
    co2ppm = 860
    lux = 5.06
    sound = 66.25

    body_state = np.array(
        [[ta_hd, tr_hd, va_hd], [ta_tr, tr_tr, va_tr], [ta_ft, tr_ft, va_ft]]
    )

    _, _, ldamdl, scale = load_hcm_model()

    assert not hcm_reduced(
        model=(ldamdl, scale),
        pre_out=pre_out,
        pre_clo=pre_clo,
        qa_ht=qa_ht,
        qa_wt=qa_wt,
        body_state=body_state,
        rh=rh,
        co2ppm=co2ppm,
        lux=lux,
        sound=sound,
    )


def test_hcm_reduced3():
    pre_out = 6
    pre_clo = 0.76
    qa_ht = 178
    qa_wt = 85
    ta_hd = 20.1
    ta_tr = 20.1
    ta_ft = 20.1
    tr_hd = 20.1
    tr_tr = 20.1
    tr_ft = 20.1
    va_hd = 0.09
    va_tr = 0.07
    va_ft = 0.08
    rh = 41.5
    co2ppm = 450
    lux = 75
    sound = 64
    light_blue = 0
    light_yellow = 0
    scent_OC = 0
    scent_Pepper = 1

    body_state = np.array(
        [[ta_hd, tr_hd, va_hd], [ta_tr, tr_tr, va_tr], [ta_ft, tr_ft, va_ft]]
    )

    _, _, ldamdl, scale = load_hcm_model()

    assert not hcm_reduced(
        model=(ldamdl, scale),
        pre_out=pre_out,
        pre_clo=pre_clo,
        qa_ht=qa_ht,
        qa_wt=qa_wt,
        body_state=body_state,
        rh=rh,
        co2ppm=co2ppm,
        lux=lux,
        sound=sound,
        light_blue=light_blue,
        light_yellow=light_yellow,
        scent_OC=scent_OC,
        scent_Pepper=scent_Pepper,
    )
