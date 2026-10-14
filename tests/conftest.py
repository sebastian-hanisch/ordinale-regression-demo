"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

m = 2, gamma = 1: u = w x1 + (1 - w) x2 mit w in [0, 1] (w = d1, d2 = 1 - w); normierte Werte a = (1; 0), b = (0,6; 0,6), c = (0,2; 0,9), d = (0,1; 0,1).
Referenzurteil: c vor b, also 0,3 - 0,7 w >= eps (w <= 3/7). Maximin: w = 0, eps = 0,3.
u(i) - u(j) ist linear in w, der kleinste Wert über w in [0, 3/7] liegt an einem Rand:
a - b = w - 0,6 (min -0,6); a - c = 1,7 w - 0,9 (min -0,9); a - d = w - 0,1 (min -0,1);
b - a = 0,6 - w (min 0,6 - 3/7); b - c = 0,7 w - 0,3 (min -0,3); b - d = 0,5;
c - a = 0,9 - 1,7 w (min 0,9 - 1,7 * 3/7); c - b = 0,3 - 0,7 w (min 0); c - d = 0,8 - 0,7 w (min 0,8 - 0,7 * 3/7);
d - a = 0,1 - w (min 0,1 - 3/7); d - b = -0,5; d - c = 0,7 w - 0,8 (min -0,8).
Notwendig: b vor a, b vor d, c vor a, c vor b, c vor d. Offen: a gegen d.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MINI_V = np.array([[1.0, 0.0], [0.6, 0.6], [0.2, 0.9], [0.1, 0.1]])
MINI_PAIRS = [(2, 1)]                                              # c vor b

# MINI_MIN[i, j] = min (u(i) - u(j)); Zeilen und Spalten a b c d
MINI_MIN = np.array([
    [0.0, -0.6, -0.9, -0.1],
    [0.6 - 3 / 7, 0.0, -0.3, 0.5],
    [0.9 - 1.7 * 3 / 7, 0.0, 0.0, 0.8 - 0.7 * 3 / 7],
    [0.1 - 3 / 7, -0.5, -0.8, 0.0],
])


@pytest.fixture
def mini_V():
    return MINI_V.copy()
