"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12): r = 0 und 1, r = n, n - r = 2, Gleichstand der Referenzen, unverträgliche Ordnung, m = 1, gleiche Alternativen."""

import numpy as np
import pytest

import ord_constants as C
import ord_evaluation as ev
import ord_uta as ut


def test_no_reference_and_one_reference_give_no_judgment():
    for r in (0, 1):
        a = ev.analyse(ev.Settings(n=8, r=r, seed=2))
        assert a.case.pairs == [] and a.case.compatible
        # ohne Urteil ist nur notwendig, was für jede nichtfallende Funktion gilt: die Dominanz (Pareto)
        V = a.case.V[a.case.others]
        dom = np.array([[bool((V[i] >= V[j]).all() and (V[i] > V[j]).any()) for j in range(len(V))] for i in range(len(V))])
        assert (a.case.nec == dom).all()


def test_the_number_of_references_is_clipped_to_leave_two_alternatives():
    assert ev.effective_r(8, 12) == 6 and ev.effective_r(8, 6) == 6 and ev.effective_r(8, 2) == 2 and ev.effective_r(3, 12) == 1
    a = ev.analyse(ev.Settings(n=8, r=12, seed=3))
    assert a.r_eff == 6 and len(a.case.others) == 2 and a.metrics["n_pairs"] <= 1


def test_ties_among_the_references_give_no_constraint():
    V = np.array([[0.2, 0.9], [0.6, 0.6], [0.5, 0.5]])
    from ord_judgments import reference_pairs

    pairs = reference_pairs(np.array([1.0, 1.0, 0.5]), [0, 1, 2])
    assert pairs == [(1, 2)]                                           # der Gleichstand 0 gegen 1 liefert keine strenge Ordnung; 1 vor 2 bleibt
    assert ut.is_compatible(ut.uta_maximin(ut.basis_coefficients(V, 1), pairs)[0])


def test_incompatible_order_means_no_relations():
    V = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.3, 0.3]])
    Cm = ut.basis_coefficients(V, 1)
    eps, _ = ut.uta_maximin(Cm, [(0, 2), (2, 1), (1, 0)])
    assert not ut.is_compatible(eps)
    region = ut.make_region(Cm, [(0, 2), (2, 1), (1, 0)])
    assert ut.min_differences(Cm, region, [0, 1, 2, 3]) is None


def test_one_criterion_every_function_is_the_same_single_ramp():
    V = np.array([[0.2], [0.9], [0.5], [0.7]])
    Cm = ut.basis_coefficients(V, 1)
    region = ut.make_region(Cm, [(1, 2)])
    nec = ut.necessary(ut.min_differences(Cm, region, [0, 1, 2, 3]))
    assert nec[1, 2] and nec[3, 0] and nec[1, 3] and nec[2, 0]          # u = d * x: die Ordnung der Werte steht fest
    assert nec.sum() == 6


def test_identical_alternatives_are_never_strictly_necessary():
    V = np.array([[0.4, 0.4], [0.4, 0.4], [0.9, 0.9]])
    Cm = ut.basis_coefficients(V, 3)
    nec = ut.necessary(ut.min_differences(Cm, ut.make_region(Cm, []), [0, 1, 2]))
    assert not nec[0, 1] and not nec[1, 0] and nec[2, 0] and nec[2, 1]


def test_extreme_settings_run():
    a = ev.analyse(ev.Settings(n=20, m=8, r=12, gamma=3, sigma=0.2, decider="untergrenze", seed=9))
    assert len(a.case.others) == 8 and a.marginals.shape == (8, 4)
    b = ev.analyse(ev.Settings(n=8, m=3, r=2, gamma=1, sigma=0.0, decider="guete_cobb", seed=1))
    assert b.case.compatible and b.marginals.shape == (3, 2)


def test_epsilon_threshold_is_the_documented_constant():
    assert C.EPS_COMPATIBLE == 1e-7 and C.TOL_NECESSARY == 1e-9
