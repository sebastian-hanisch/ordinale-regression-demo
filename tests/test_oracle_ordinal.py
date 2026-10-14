"""Orakel: UTA und robuste Regression gegen unabhängige Rechenwege (DEMO-PLAYBOOK §12).

1. Modellklasse: Entscheider mit Rohwerten (additiv, linear), widerspruchsfrei: alle notwendigen Paare stimmen (gamma = 1, 2, 3); Gegenprobe: ein nicht additiver Entscheider erzeugt falsche.
2. Notwendig ist streng antisymmetrisch und impliziert möglich; das Referenzurteil selbst ist notwendig.
3. Monotonie in den Beispielen: mehr Referenzurteile (dieselbe Ordnung verlängert) entscheiden nie weniger Paare.
4. m = 2, gamma = 1 exakt: min (u(a) - u(b)) gegen die Gewichte auf einem feinen Gitter.
5. Gesampelt gegen ausgelotet: jede Alternative, die in verträglichen Ziehungen Erste ist, ist per LP möglich beste.
"""

import numpy as np
import pytest

import ord_constants as C
import ord_uta as ut

SEEDS = tuple(range(300000, 300030))
N, M = 9, 4


def world(seed, n=N, m=M):
    rng = np.random.default_rng(seed)
    V = rng.random((n, m))
    w = rng.dirichlet(np.ones(m))
    return V, w


def ordered_pairs(u, refs):
    order = sorted(refs, key=lambda i: (-u[i], i))
    return [(a, b) for a, b in zip(order[:-1], order[1:]) if u[a] > u[b]]


def relations(V, u, refs, gamma):
    others = [i for i in range(len(V)) if i not in refs]
    Cm = ut.basis_coefficients(V, gamma)
    region = ut.make_region(Cm, ordered_pairs(u, refs))
    return others, ut.min_differences(Cm, region, others)


def test_in_the_model_class_every_necessary_pair_is_right():
    wrong = total = 0
    for seed in SEEDS:
        V, w = world(seed)
        u = V @ w
        refs = sorted(np.random.default_rng(seed).choice(N, size=4, replace=False).tolist())
        for gamma in (1, 2, 3):
            others, mn = relations(V, u, refs, gamma)
            assert mn is not None                                      # die Ordnung ist verträglich (die wahre Funktion liegt in der Klasse)
            nec = ut.necessary(mn)
            for i in range(len(others)):
                for j in range(len(others)):
                    if nec[i, j]:
                        total += 1
                        wrong += int(u[others[i]] <= u[others[j]])
    assert total > 100 and wrong == 0


def test_a_non_additive_decider_produces_wrong_necessary_pairs_in_some_cases():
    wrong = 0
    rng = np.random.default_rng(11)
    for seed in SEEDS:
        V, w = world(seed, 12, 3)
        u = np.minimum.reduce([V[:, j] / w[j] for j in range(3)])        # Leontief: das schlechteste (gewichtete) Kriterium zählt, nicht additiv
        refs = sorted(rng.choice(12, size=5, replace=False).tolist())
        others, mn = relations(V, u, refs, 1)
        if mn is None:
            continue
        nec = ut.necessary(mn)
        wrong += sum(int(u[others[i]] <= u[others[j]]) for i in range(len(others)) for j in range(len(others)) if nec[i, j])
    assert wrong > 0


def test_necessary_is_antisymmetric_and_implies_possible_and_the_judgment_holds():
    for seed in SEEDS:
        V, w = world(seed)
        u = V @ w
        refs = list(range(5))
        pairs = ordered_pairs(u, refs)
        Cm = ut.basis_coefficients(V, 2)
        region = ut.make_region(Cm, pairs)
        mn = ut.min_differences(Cm, region, list(range(N)))
        nec, pos = ut.necessary(mn), ut.possible(mn)
        assert not (nec & nec.T).any()
        assert (pos | ~nec).all()                                      # notwendig => möglich
        for a, b in pairs:
            assert nec[a, b]


def test_more_judgments_never_decide_fewer_pairs():
    for seed in SEEDS[:15]:
        V, w = world(seed)
        u = V @ w
        ranked = sorted(range(N), key=lambda i: -u[i])
        Cm = ut.basis_coefficients(V, 1)
        tail = list(range(N))
        previous = None
        for r in (2, 4, 6):
            refs = ranked[:r]
            pairs = [(a, b) for a, b in zip(refs[:-1], refs[1:])]
            mn = ut.min_differences(Cm, ut.make_region(Cm, pairs), tail)
            nec = ut.necessary(mn)
            if previous is not None:
                assert (nec | ~previous).all()                         # was vorher notwendig war, bleibt es
            previous = nec


def test_two_criteria_match_a_grid_over_the_weights():
    rng = np.random.default_rng(5)
    for _ in range(15):
        V = rng.random((5, 2))
        u = V @ np.array([0.6, 0.4])
        refs = [0, 1, 2]
        pairs = ordered_pairs(u, refs)
        Cm = ut.basis_coefficients(V, 1)
        mn = ut.min_differences(Cm, ut.make_region(Cm, pairs), [3, 4])
        grid = np.linspace(0, 1, 40001)
        W = np.column_stack([grid, 1 - grid])
        ok = np.ones(len(W), dtype=bool)
        for a, b in pairs:
            ok &= (W @ (V[a] - V[b])) >= C.EPS_COMPATIBLE
        d = W[ok] @ (V[3] - V[4])
        assert mn[0, 1] == pytest.approx(d.min(), abs=1e-3)
        assert mn[1, 0] == pytest.approx((-d).min(), abs=1e-3)


def test_every_alternative_that_is_first_in_compatible_draws_is_possible_best_by_lp():
    for seed in SEEDS[:10]:
        V, w = world(seed, 10, 4)
        u = V @ w
        pairs = ordered_pairs(u, [0, 1, 2, 3, 4])
        rng = np.random.default_rng(seed)
        W = rng.dirichlet(np.ones(4), 20000)
        ok = ut.share_compatible(V, pairs, W)
        poss = ut.possible_best_in_region(V, pairs)
        top = np.bincount((W[ok] @ V.T).argmax(axis=1), minlength=10) > 0
        assert not (top & ~poss).any()
        assert poss[int(np.argmax(u))]                                  # die wahren Gewichte sind zulässig: der Beste ist möglich
