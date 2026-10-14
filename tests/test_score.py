"""Nenn-Spannweite, Nutzen der Entscheider und Kendalls tau-b."""

import numpy as np
import pytest

import ord_score as sc


def test_nominal_normalisation_uses_the_fixed_ranges():
    lo, hi = sc.nominal_bounds(3)
    assert sc.normalize_nominal(np.array([lo, hi, (lo + hi) / 2])).ravel() == pytest.approx(np.repeat([0.0, 1.0, 0.5], 3))


def test_decider_utilities_by_hand():
    G = np.array([[0.04, 0.9], [0.5, 0.5]])
    S = np.array([[2000.0, 0.0], [0.0, 0.0]])
    w = np.array([0.5, 0.5])
    assert sc.decider_utility("guete_linear", G, S, w).tolist() == pytest.approx([0.47, 0.5])
    assert sc.decider_utility("guete_konkav", G, S, w).tolist() == pytest.approx([0.5 * (0.2 + np.sqrt(0.9)), 0.5 * np.sqrt(0.5) * 2])
    u = sc.decider_utility("untergrenze", G, S, w)
    assert u[0] < -1e8 and u[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        sc.decider_utility("orakel", G, S, w)


def test_the_raw_value_decider_is_linear_on_the_nominal_range():
    lo, hi = sc.nominal_bounds(2)
    S = np.array([lo, hi, (lo + hi) / 2])
    assert sc.decider_utility("roh_linear", np.zeros((3, 2)), S, np.array([0.7, 0.3])).tolist() == pytest.approx([0.0, 1.0, 0.5])


def test_kendall_tau_b():
    assert sc.kendall_tau_b(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(1.0)
    assert sc.kendall_tau_b(np.array([1, 2, 3]), np.array([3, 2, 1])) == pytest.approx(-1.0)
    assert sc.kendall_tau_b(np.array([1, 1, 1]), np.array([1, 2, 3])) == 0.0
