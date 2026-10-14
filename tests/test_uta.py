"""UTA und robuste Regression gegen die Handrechnung der Mini-Instanz (tests/conftest.py) und kleine Festwerte."""

import numpy as np
import pytest

import ord_constants as C
import ord_uta as ut
from conftest import MINI_MIN, MINI_PAIRS, MINI_V


def test_basis_coefficients_by_hand():
    V = np.array([[0.75, 0.0, 1.0]])
    assert ut.basis_coefficients(V, 1).tolist() == [[0.75, 0.0, 1.0]]
    assert ut.basis_coefficients(V, 2).tolist() == [[1.0, 0.5, 0.0, 0.0, 1.0, 1.0]]            # Abschnitte [0; 0,5] und [0,5; 1] je Kriterium
    assert ut.basis_coefficients(np.array([[0.1]]), 4).ravel() == pytest.approx([0.4, 0.0, 0.0, 0.0])
    assert ut.basis_coefficients(np.array([[1.5]]), 2).tolist() == [[1.0, 1.0]]               # über der Nenn-Spannweite: gesättigt


def test_maximin_on_the_mini_instance_is_w_zero_and_eps_zero_point_three():
    Cm = ut.basis_coefficients(MINI_V, 1)
    eps, d = ut.uta_maximin(Cm, MINI_PAIRS)
    assert eps == pytest.approx(0.3) and d == pytest.approx([0.0, 1.0])
    assert ut.is_compatible(eps)


def test_incompatible_order_gives_a_nonpositive_epsilon():
    V = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    # a vor c und c vor b und b vor a: ein Zyklus kann keine Funktion erfüllen
    eps, _ = ut.uta_maximin(ut.basis_coefficients(V, 1), [(0, 2), (2, 1), (1, 0)])
    assert eps <= 1e-9 and not ut.is_compatible(eps)


def test_no_pairs_gives_equal_increments_and_a_compatible_epsilon():
    Cm = ut.basis_coefficients(MINI_V, 3)
    eps, d = ut.uta_maximin(Cm, [])
    assert ut.is_compatible(eps) and d == pytest.approx(np.full(6, 1 / 6))


def test_minimum_differences_on_the_mini_instance(mini_V):
    Cm = ut.basis_coefficients(mini_V, 1)
    region = ut.make_region(Cm, MINI_PAIRS)
    mn = ut.min_differences(Cm, region, [0, 1, 2, 3])
    assert mn == pytest.approx(MINI_MIN, abs=1e-5)


def test_necessary_and_possible_on_the_mini_instance(mini_V):
    Cm = ut.basis_coefficients(mini_V, 1)
    mn = ut.min_differences(Cm, ut.make_region(Cm, MINI_PAIRS), [0, 1, 2, 3])
    nec = ut.necessary(mn)
    expected = np.zeros((4, 4), dtype=bool)
    for i, j in ((1, 0), (1, 3), (2, 0), (2, 1), (2, 3)):             # b vor a, b vor d, c vor a, c vor b, c vor d
        expected[i, j] = True
    assert (nec == expected).all()
    pos = ut.possible(mn)
    assert pos[0, 3] and pos[3, 0]                                    # a gegen d: beide Richtungen möglich
    assert not pos[0, 1] and not pos[0, 2] and not pos[3, 1] and pos[1, 0] and pos[2, 1]
    assert pos.diagonal().all()


def test_rank_interval_on_the_mini_instance(mini_V):
    Cm = ut.basis_coefficients(mini_V, 1)
    nec = ut.necessary(ut.min_differences(Cm, ut.make_region(Cm, MINI_PAIRS), [0, 1, 2, 3]))
    best, worst = ut.rank_interval(nec)
    assert best.tolist() == [3, 2, 1, 3] and worst.tolist() == [4, 2, 1, 4]


def test_random_vertex_is_compatible_and_a_corner():
    Cm = ut.basis_coefficients(MINI_V, 1)
    region = ut.make_region(Cm, MINI_PAIRS)
    from entscheidungsmatrix_generator import SplitMix64

    seen = set()
    rng = SplitMix64(3)
    for _ in range(12):
        d = ut.random_vertex(region, rng)
        assert d.sum() == pytest.approx(1.0) and (d >= -1e-12).all()
        assert 0.3 - 0.7 * d[0] >= C.EPS_COMPATIBLE - 1e-12           # c vor b
        seen.add(round(float(d[0]), 4))
    assert seen <= {0.0, round((0.3 - C.EPS_COMPATIBLE) / 0.7, 4)}    # nur die beiden Ecken der Strecke


def test_marginals_accumulate_the_increments():
    d = np.array([0.1, 0.2, 0.3, 0.4])                                # m = 2, gamma = 2
    assert ut.marginals(d, 2, 2).ravel() == pytest.approx([0.0, 0.1, 0.3, 0.0, 0.3, 0.7])


def test_weight_region_share_and_possible_best_on_the_mini_instance(mini_V):
    W = np.array([[0.0, 1.0], [0.2, 0.8], [0.42, 0.58], [0.5, 0.5], [1.0, 0.0]])
    assert ut.share_compatible(mini_V, MINI_PAIRS, W).tolist() == [True, True, True, False, False]       # w <= 3/7 = 0,4286
    assert ut.possible_best_in_region(mini_V, MINI_PAIRS).tolist() == [False, False, True, False]     # w <= 3/7: c ist beste, b erst ab 3/7 (Gleichstand), a erst ab 0,6


def test_weak_region_accepts_draws_where_a_pair_is_exactly_tied():
    V = np.array([[0.5, 0.5], [0.5, 0.5], [0.2, 0.9]])               # a und b sind gleich: (V_a - V_b) w = 0 für jedes w
    W = np.array([[0.3, 0.7], [0.9, 0.1]])
    assert ut.share_compatible(V, [(0, 1)], W).tolist() == [True, True]
    assert ut.share_compatible(V, [(0, 2)], W).tolist() == [False, True]       # u(a) = 0,5 gegen u(c) = 0,69 bei t = 0,3 (nein) und 0,27 bei t = 0,9 (ja)
