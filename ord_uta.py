"""UTA und robuste ordinale Regression per LP: stückweise lineare nichtfallende Randfunktionen, Ordnung der Referenzen als Nebenbedingungen.

u(a) = C[a] @ d mit d >= 0 den Zuwächsen der Randfunktionen je Abschnitt und sum(d) = 1 (u(1, ..., 1) = 1). Kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

import ord_constants as C


def basis_coefficients(V: np.ndarray, gamma: int) -> np.ndarray:
    """n x (m gamma): Anteil des Abschnitts k am Wert von Kriterium j (Spalte j gamma + k); u(a) = C[a] @ d."""
    V = np.asarray(V, dtype=float)
    n, m = V.shape
    ks = np.arange(gamma)
    return np.clip(gamma * V[:, :, None] - ks[None, None, :], 0.0, 1.0).reshape(n, m * gamma)


def _compat_rows(Cm: np.ndarray, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """-(C_a - C_b) d <= -EPS: a muss mindestens um EPS vor b liegen."""
    gm = Cm.shape[1]
    if not pairs:
        return np.zeros((0, gm)), np.zeros(0)
    A = -np.array([Cm[a] - Cm[b] for a, b in pairs])
    return A, np.full(len(pairs), -C.EPS_COMPATIBLE)


def uta_maximin(Cm: np.ndarray, pairs: list[tuple[int, int]]) -> tuple[float, np.ndarray]:
    """Maximiere epsilon mit (C_a - C_b) d >= epsilon für alle Paare. epsilon > EPS_COMPATIBLE: die Ordnung ist streng reproduzierbar; sonst nicht
    (epsilon <= 0: auch schwach nicht). Ohne Paare: gleichmäßige Zuwächse."""
    gm = Cm.shape[1]
    if not pairs:
        return 1.0, np.full(gm, 1.0 / gm)
    rows = np.array([np.concatenate([-(Cm[a] - Cm[b]), [1.0]]) for a, b in pairs])
    cost = np.zeros(gm + 1)
    cost[-1] = -1.0
    res = linprog(cost, A_ub=rows, b_ub=np.zeros(len(pairs)), A_eq=np.concatenate([np.ones(gm), [0.0]])[None, :], b_eq=[1.0],
                  bounds=[(0, None)] * gm + [(None, 1.0)], method="highs")
    return float(res.x[-1]), res.x[:gm]


def is_compatible(eps: float) -> bool:
    return eps > C.EPS_COMPATIBLE


@dataclass
class Region:
    """Menge der verträglichen Funktionen: d >= 0, sum d = 1, jedes Paar mindestens EPS_COMPATIBLE Abstand."""
    A: np.ndarray
    b: np.ndarray
    gm: int

    def minimize(self, c: np.ndarray):
        """(Wert, d) des Minimums von c @ d über die Region; (None, None), wenn die Region leer ist."""
        res = linprog(c, A_ub=self.A if len(self.b) else None, b_ub=self.b if len(self.b) else None, A_eq=np.ones((1, self.gm)), b_eq=[1.0],
                      bounds=[(0, None)] * self.gm, method="highs")
        return (float(res.fun), res.x) if res.status == 0 else (None, None)


def make_region(Cm: np.ndarray, pairs: list[tuple[int, int]]) -> Region:
    A, b = _compat_rows(Cm, pairs)
    return Region(A, b, Cm.shape[1])


def min_differences(Cm: np.ndarray, region: Region, idx: list[int]) -> np.ndarray | None:
    """mn[i, j] = min (u(idx[i]) - u(idx[j])) über die Region; None, wenn die Region leer ist (Ordnung nicht verträglich)."""
    k = len(idx)
    mn = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            v, _ = region.minimize(Cm[idx[i]] - Cm[idx[j]])
            if v is None:
                return None
            mn[i, j] = v
    return mn


def necessary(mn: np.ndarray) -> np.ndarray:
    """nec[i, j]: i ist für alle verträglichen Funktionen mindestens so gut wie j, und für j gilt das nicht (streng notwendig)."""
    ge = mn >= -C.TOL_NECESSARY
    nec = ge & ~ge.T
    np.fill_diagonal(nec, False)
    return nec


def possible(mn: np.ndarray) -> np.ndarray:
    """pos[i, j]: es gibt eine verträgliche Funktion mit u(i) >= u(j): max (u(i) - u(j)) = -min (u(j) - u(i)) >= 0."""
    pos = mn.T <= C.TOL_NECESSARY
    np.fill_diagonal(pos, True)
    return pos


def rank_interval(nec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bester, schlechtester) Rang je Alternative (1 = beste) aus den notwendigen Relationen innerhalb der Menge."""
    k = nec.shape[0]
    return 1 + nec.sum(axis=0), k - nec.sum(axis=1)


def random_vertex(region: Region, rng) -> np.ndarray | None:
    """Eine zufällige Ecke der Region: Minimum einer zufälligen Richtung. Eine einzelne, willkürliche verträgliche Funktion."""
    _, d = region.minimize(rng.normal(region.gm))
    return d


def marginals(d: np.ndarray, m: int, gamma: int) -> np.ndarray:
    """m x (gamma + 1): Wert der Randfunktion je Kriterium an den Stützstellen 0, 1/gamma, ..., 1."""
    inc = np.asarray(d, dtype=float).reshape(m, gamma)
    return np.hstack([np.zeros((m, 1)), np.cumsum(inc, axis=1)])


# ----------------------------------------------------------------------------- Kontrast: Gewichtsraum gesampelt gegen ausgelotet (nur gamma = 1)

def share_compatible(V: np.ndarray, pairs: list[tuple[int, int]], W: np.ndarray) -> np.ndarray:
    """Maske der Ziehungen W (ns x m), die jedes Paar schwach erfüllen: (V_a - V_b) w >= 0."""
    ok = np.ones(len(W), dtype=bool)
    V = np.asarray(V, dtype=float)
    for a, b in pairs:
        ok &= (np.asarray(W, dtype=float) @ (V[a] - V[b])) >= 0
    return ok


def possible_best_in_region(V: np.ndarray, pairs: list[tuple[int, int]], tol: float = 1e-9) -> np.ndarray:
    """Exakt per LP: Alternativen, die für ein Gewicht mit erfüllten Paaren (schwach) streng beste sind."""
    V = np.asarray(V, dtype=float)
    n, m = V.shape
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        D = V[i] - np.delete(V, i, axis=0)
        A = np.hstack([-D, np.ones((n - 1, 1))])
        b = np.zeros(n - 1)
        if pairs:
            R = np.array([np.concatenate([-(V[a] - V[bb]), [0.0]]) for a, bb in pairs])
            A = np.vstack([A, R])
            b = np.concatenate([b, np.zeros(len(pairs))])
        cost = np.zeros(m + 1)
        cost[-1] = -1.0
        res = linprog(cost, A_ub=A, b_ub=b, A_eq=np.concatenate([np.ones(m), [0.0]])[None, :], b_eq=[1.0], bounds=[(0, None)] * m + [(None, None)], method="highs")
        out[i] = res.status == 0 and -res.fun > tol
    return out
