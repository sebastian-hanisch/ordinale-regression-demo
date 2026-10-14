"""Nenn-Spannweite, Nutzen der Entscheider, Kendalls tau-b. Nur numpy; eigene Kopie der kleinen Helfer (kein Import über Repos).
Alle Funktionen arbeiten im vorzeichenbehafteten Raum, in dem HÖHER BESSER gilt."""

from __future__ import annotations

import numpy as np

import ord_constants as C
from entscheidungsmatrix_generator import CRITERIA, utility


def nominal_bounds(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Nenn-Spannweite je Kriterium im vorzeichenbehafteten Raum: ↑ [lo, hi], ↓ [-hi, -lo]."""
    lo = np.array([CRITERIA[k][3] if CRITERIA[k][1] > 0 else -CRITERIA[k][4] for k in range(m)], dtype=float)
    hi = np.array([CRITERIA[k][4] if CRITERIA[k][1] > 0 else -CRITERIA[k][3] for k in range(m)], dtype=float)
    return lo, hi


def normalize_nominal(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    lo, hi = nominal_bounds(S.shape[1])
    return (S - lo) / (hi - lo)


def decider_utility(kind: str, G: np.ndarray, S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Nutzen je Alternative für die fünf Entscheider (G Güte, S signierte Rohwerte). Untergrenze: ein Kriterium unter Güte THETA gibt UNDER_FLOOR."""
    if kind == "roh_linear":
        return normalize_nominal(S) @ w
    forms = {"guete_linear": "linear", "guete_konkav": "konkav", "guete_cobb": "cobb", "untergrenze": "linear"}
    if kind not in forms:
        raise ValueError(f"unbekannter Entscheider: {kind}")
    u = utility(np.asarray(G, dtype=float), w, forms[kind])
    if kind == "untergrenze":
        u = np.where((np.asarray(G) >= C.THETA).all(axis=1), u, C.UNDER_FLOOR)
    return u


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)
