"""Beispielurteile: Auswahl der Referenzen, Rauschen, Paare aufeinanderfolgender Referenzen."""

import numpy as np
import pytest

import ord_judgments as jd
from entscheidungsmatrix_generator import SplitMix64


def test_references_are_distinct_sorted_and_deterministic():
    refs = jd.choose_references(12, 5, SplitMix64(3))
    assert len(refs) == 5 and len(set(refs)) == 5 and refs == sorted(refs) and all(0 <= i < 12 for i in refs)
    assert refs == jd.choose_references(12, 5, SplitMix64(3))
    assert refs != jd.choose_references(12, 5, SplitMix64(4))


def test_reference_count_is_clipped_to_the_number_of_alternatives():
    assert jd.choose_references(4, 9, SplitMix64(1)) == [0, 1, 2, 3]
    assert jd.choose_references(4, 0, SplitMix64(1)) == []
    assert jd.choose_references(4, -3, SplitMix64(1)) == []


def test_reference_choice_is_spread_over_all_alternatives():
    counts = np.zeros(10)
    for seed in range(400):
        for i in jd.choose_references(10, 3, SplitMix64(seed)):
            counts[i] += 1
    assert counts.min() > 80 and counts.max() < 160                    # Erwartung 120 je Alternative


def test_observed_utility_without_noise_is_the_utility_and_noise_keeps_the_sign():
    u = np.array([0.5, -0.2, 0.9])
    assert jd.observed_utility(u, 0.0, SplitMix64(1)).tolist() == u.tolist()
    noisy = jd.observed_utility(u, 0.2, SplitMix64(1))
    assert (np.sign(noisy) == np.sign(u)).all() and not np.allclose(noisy, u)
    assert jd.observed_utility(u, 0.0, SplitMix64(1)) is not u


def test_noise_has_the_documented_lognormal_spread():
    z = np.log(jd.observed_utility(np.ones(40000), 0.2, SplitMix64(7)))
    assert z.mean() == pytest.approx(0.0, abs=0.01) and z.std() == pytest.approx(0.2, abs=0.01)


def test_reference_pairs_follow_the_order_and_skip_ties():
    u = np.array([0.3, 0.9, 0.5, 0.5, 0.1])
    assert jd.reference_pairs(u, [0, 1, 2, 4]) == [(1, 2), (2, 0), (0, 4)]
    assert jd.reference_pairs(u, [2, 3]) == []                         # gleicher Nutzen: kein Urteil
    assert jd.reference_pairs(u, [1]) == [] and jd.reference_pairs(u, []) == []
