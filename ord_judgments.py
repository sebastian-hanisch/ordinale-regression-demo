"""Beispielurteile des simulierten Entscheiders: Referenzen wählen, ihre Ordnung (mit Rauschen) und die Paare aufeinanderfolgender Referenzen. Zufall aus SplitMix64."""

from __future__ import annotations

import numpy as np

from entscheidungsmatrix_generator import SplitMix64


def choose_references(n: int, r: int, rng: SplitMix64) -> list[int]:
    """r verschiedene Alternativen aus n, aufsteigend nach Nummer (zufällige Auswahl, kein Wissen über die Nutzen)."""
    r = int(max(0, min(r, n)))
    return sorted(rng.permutation(n)[:r].tolist())


def observed_utility(u: np.ndarray, sigma: float, rng: SplitMix64) -> np.ndarray:
    """Der Nutzen, nach dem der Entscheider urteilt: u * exp(sigma Z) (sigma = 0: genau sein Nutzen). Das Vorzeichen bleibt, die Ordnung wird nur gestört."""
    u = np.asarray(u, dtype=float)
    if sigma == 0:
        return u.copy()
    return u * np.exp(sigma * rng.normal(len(u)))


def reference_pairs(u: np.ndarray, refs: list[int]) -> list[tuple[int, int]]:
    """Paare (a, b) aufeinanderfolgender Referenzen in der Ordnung des Entscheiders (a vor b); bei gleichem Nutzen kein Paar."""
    order = sorted(refs, key=lambda i: (-u[i], i))
    return [(a, b) for a, b in zip(order[:-1], order[1:]) if u[a] > u[b]]
