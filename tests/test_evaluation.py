"""Auswertung: Fall der App, Grenzfälle und Experimente auf wenigen Seeds (Aussagen über die festen Matrizen stehen in test_claims.py)."""

import numpy as np
import pytest

import ord_constants as C
import ord_evaluation as ev


def test_de_uses_a_decimal_comma():
    assert ev.de(3.14159, 2) == "3,14"
    assert ev.de(2.0, 0) == "2"


def test_analysis_of_the_default_case_is_consistent():
    a = ev.analyse(ev.Settings())
    s, c = a.settings, a.case
    assert a.r_eff == min(s.r, s.n - 2) and len(c.refs) == a.r_eff and len(c.others) == s.n - a.r_eff
    assert sorted(c.refs + c.others) == list(range(s.n))
    assert c.compatible and c.mn is not None and c.nec.shape == (len(c.others), len(c.others))
    assert not (c.nec & c.nec.T).any()
    assert a.marginals.shape == (s.m, s.gamma + 1) and a.marginals[:, 0].tolist() == [0.0] * s.m
    assert a.marginals[:, -1].sum() == pytest.approx(1.0)
    assert sorted(a.true_rank.tolist()) == list(range(1, len(c.others) + 1))


def test_the_same_settings_give_the_same_analysis():
    a, b = ev.analyse(ev.Settings(seed=7)), ev.analyse(ev.Settings(seed=7))
    assert a.case.refs == b.case.refs and (a.case.u_hat == b.case.u_hat).all() and a.metrics == b.metrics


def test_the_reference_order_is_that_of_the_decider_without_noise():
    a = ev.analyse(ev.Settings(seed=3, sigma=0.0))
    u = a.case.utility
    order = sorted(a.case.refs, key=lambda i: (-u[i], i))
    assert a.case.pairs == list(zip(order[:-1], order[1:]))


def test_noise_can_make_the_order_incompatible_and_removes_the_relations():
    found = False
    for seed in range(1, 30):
        a = ev.analyse(ev.Settings(sigma=0.2, seed=seed))
        if not a.case.compatible:
            found = True
            assert a.case.nec is None and a.metrics["decided"] is None and a.best_rank is None
            break
    assert found


def test_a_non_additive_decider_can_give_a_wrong_necessary_pair():
    assert any(ev.analyse(ev.Settings(decider="untergrenze", seed=s)).metrics["wrong_n"] for s in range(1, 40))


def test_flexible_functions_decide_no_more_pairs_than_weights_only():
    for seed in range(1, 8):
        w1 = ev.analyse(ev.Settings(gamma=1, seed=seed)).metrics["decided"]
        w3 = ev.analyse(ev.Settings(gamma=3, seed=seed)).metrics["decided"]
        assert w3 <= w1 + 1e-12                                          # die Klasse mit 3 Abschnitten enthält die mit einem: mehr Funktionen, weniger notwendig


def test_in_the_model_class_every_necessary_pair_of_the_analysis_is_right():
    for seed in range(1, 10):
        for gamma in C.GAMMA_CHOICES:
            m = ev.analyse(ev.Settings(gamma=gamma, seed=seed)).metrics
            assert m["compatible"] and m["wrong_n"] == 0 and m["best_possible"]


def test_experiments_accept_a_small_set_of_seeds():
    seeds = C.SWEEP_SEEDS[:3]
    ex = ev.experiment_examples(rs=(3, 6), seeds=seeds)
    assert len(ex) == 4 and all(0.0 <= r["decided"] <= 1.0 and r["correct"] == 1.0 for r in ex)
    md = ev.experiment_model(deciders=("roh_linear", "untergrenze"), sigmas=(0.2,), gammas=(1,), seeds=seeds)
    assert [r["kind"] for r in md] == ["decider", "decider", "noise"]
    ar = ev.experiment_arbitrary(rs=(4,), gammas=(1, 3), seeds=seeds, k_random=3)
    assert len(ar) == 2 and all(r["maximin"] is None or 0.0 <= r["maximin"] <= 1.0 for r in ar)
    sa = ev.experiment_sampling(rs=(3, 6), seeds=seeds, draws=2000)
    assert len(sa) == 2 and sa[0]["share"] > sa[1]["share"] and all(r["in_region"] == 3 for r in sa)


def test_the_share_of_compatible_draws_falls_with_every_further_judgment():
    sa = ev.experiment_sampling(rs=(2, 4, 6), seeds=C.SWEEP_SEEDS[:6], draws=20000)
    shares = [r["share"] for r in sa]
    assert shares[0] > shares[1] > shares[2]
    assert np.all(np.array([r["possible_lp"] for r in sa])[:-1] >= np.array([r["possible_lp"] for r in sa])[1:])
