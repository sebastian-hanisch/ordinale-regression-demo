"""Jede Zahl der README, gemessen mit den eigenen Experimenten der Demo (feste Matrizen, SWEEP_SEEDS). Die Werte sind deterministisch (SplitMix64)."""

import pytest

import ord_constants as C
import ord_evaluation as ev


@pytest.fixture(scope="module")
def exp():
    return {"examples": ev.experiment_examples(), "model": ev.experiment_model(), "arbitrary": ev.experiment_arbitrary(), "sampling": ev.experiment_sampling()}


def _row(rows, **kw):
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


def test_how_many_judgments_until_possible_becomes_necessary(exp):
    rows = exp["examples"]
    assert [_row(rows, r=r, gamma=1)["decided"] for r in C.EXP_RS] == pytest.approx([0.132, 0.304, 0.512, 0.542], abs=0.0006)
    assert [_row(rows, r=r, gamma=3)["decided"] for r in C.EXP_RS] == pytest.approx([0.062, 0.095, 0.131, 0.146], abs=0.0006)
    assert all(r["correct"] == 1.0 and r["compatible"] == 1.0 and r["best_possible"] == 1.0 for r in rows)
    assert [_row(rows, r=r, gamma=1)["width"] for r in C.EXP_RS] == pytest.approx([6.944, 4.179, 1.95, 0.917], abs=0.0006)
    assert [_row(rows, r=r, gamma=3)["width"] for r in C.EXP_RS] == pytest.approx([7.5, 5.429, 3.475, 1.708], abs=0.0006)
    assert [_row(rows, r=r, gamma=1)["possible_best"] for r in C.EXP_RS] == pytest.approx([6.25, 3.875, 2.5, 1.812], abs=0.0006)
    assert [_row(rows, r=r, gamma=1)["necessary_best"] for r in C.EXP_RS] == pytest.approx([0.0, 0.188, 0.312, 0.375], abs=0.0006)


def test_the_model_class_and_the_noise_decide_whether_necessary_is_right(exp):
    rows = exp["model"]
    ok = _row(rows, kind="decider", decider="roh_linear", gamma=1)
    assert (ok["compatible"], ok["decided"], ok["correct"], ok["wrong_n"]) == pytest.approx((1.0, 0.363, 1.0, 0.0), abs=0.0006)
    g1 = _row(rows, kind="decider", decider="guete_linear", gamma=1)
    assert (g1["correct"], g1["wrong_n"]) == pytest.approx((0.984, 0.062), abs=0.0006)
    nl = _row(rows, kind="decider", decider="untergrenze", gamma=1)
    assert (nl["compatible"], nl["decided"], nl["correct"], nl["wrong_n"]) == pytest.approx((1.0, 0.348, 0.943, 0.375), abs=0.0006)
    assert _row(rows, kind="decider", decider="untergrenze", gamma=3)["correct"] == 1.0
    assert [_row(rows, kind="noise", sigma=s, gamma=1)["compatible"] for s in (0.1, 0.2)] == pytest.approx([0.938, 0.688], abs=0.0006)
    assert [_row(rows, kind="noise", sigma=s, gamma=1)["correct"] for s in (0.1, 0.2)] == pytest.approx([0.956, 0.939], abs=0.0006)
    assert [_row(rows, kind="noise", sigma=s, gamma=1)["wrong_n"] for s in (0.1, 0.2)] == pytest.approx([0.4, 0.545], abs=0.0006)
    assert [_row(rows, kind="noise", sigma=s, gamma=3)["compatible"] for s in (0.1, 0.2)] == pytest.approx([1.0, 0.875], abs=0.0006)


def test_a_single_solution_is_a_choice(exp):
    rows = exp["arbitrary"]
    assert [_row(rows, r=r, gamma=1)["undecided"] for r in (4, 6, 8)] == pytest.approx([0.79, 0.638, 0.49], abs=0.0006)
    assert [_row(rows, r=r, gamma=1)["maximin"] for r in (4, 6, 8)] == pytest.approx([0.828, 0.785, 0.732], abs=0.0006)
    assert [_row(rows, r=r, gamma=1)["random"] for r in (4, 6, 8)] == pytest.approx([0.623, 0.679, 0.589], abs=0.0006)
    assert [_row(rows, r=r, gamma=1)["disagree"] for r in (4, 6, 8)] == pytest.approx([0.4, 0.408, 0.447], abs=0.0006)
    assert [_row(rows, r=r, gamma=3)["disagree"] for r in (4, 6, 8)] == pytest.approx([0.318, 0.322, 0.357], abs=0.0006)
    assert [_row(rows, r=r, gamma=3)["undecided"] for r in (4, 6, 8)] == pytest.approx([0.935, 0.904, 0.844], abs=0.0006)


def test_sampling_hits_the_compatible_region_ever_more_rarely(exp):
    rows = exp["sampling"]
    assert [r["share"] for r in rows] == pytest.approx([0.368, 0.196, 0.049, 0.015], abs=0.0006)
    assert [r["possible_lp"] for r in rows] == pytest.approx([5.583, 4.167, 2.5, 1.917], abs=0.0006)
    assert [r["found"] for r in rows] == pytest.approx([5.208, 3.708, 2.208, 1.75], abs=0.0006)
    assert [r["missed"] for r in rows] == pytest.approx([0.375, 0.458, 0.292, 0.167], abs=0.0006)
    assert all(r["in_region"] == r["matrices"] == 24 and r["none"] == 0 for r in rows)
    assert rows[-1]["accepted"] == pytest.approx(1494.5, abs=0.6)
