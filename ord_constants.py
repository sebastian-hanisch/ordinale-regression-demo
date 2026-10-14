"""Konstanten: Regler-Grenzen, Entscheider, Randfunktionen, Rauschen, feste Annahmen der Experimente und Presets."""

N_MIN, N_MAX, DEFAULT_N = 8, 20, 12
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5
R_MIN, R_MAX, DEFAULT_R = 2, 12, 6            # Zahl der Referenzalternativen, die der Entscheider ordnet (höchstens n - 2 wirksam)
SEED_MAX, DEFAULT_SEED = 999999, 5            # je Preset geprüft (tests/test_presets.py)

GAMMA_CHOICES = (1, 3)                        # Abschnitte je Randfunktion: 1 = nur Gewichte, 3 = stückweise linear
DEFAULT_GAMMA = 1
GAMMA_LABELS = {1: "1 Abschnitt (nur Gewichte: linear in den Werten)", 3: "3 Abschnitte (stückweise lineare Randfunktionen)"}
SIGMA_CHOICES = (0.0, 0.1, 0.2)               # Rauschen der Urteile: Nutzen mal exp(sigma Z)
DEFAULT_SIGMA = 0.0

DECIDERS = ("guete_linear", "guete_konkav", "guete_cobb", "roh_linear", "untergrenze")
DECIDER_LABELS = {
    "guete_linear": "Linear in der Güte (Quantil innerhalb des Kriteriums): nichtlineare Randfunktion",
    "guete_konkav": "Konkav in der Güte (Wurzel, gewichtet): nichtlineare Randfunktion",
    "guete_cobb": "Cobb-Douglas in der Güte (Logarithmus, gewichtet): nichtlineare Randfunktion",
    "roh_linear": "Linear in den Rohwerten (feste Nenn-Spannweite): liegt in der Modellklasse",
    "untergrenze": "Linear in der Güte, mit Untergrenze (kein Kriterium unter 0,1): nicht additiv",
}
DECIDER_SHORT = {"guete_linear": "Güte/linear", "guete_konkav": "Güte/konkav", "guete_cobb": "Güte/Cobb-Douglas", "roh_linear": "Rohwerte/linear", "untergrenze": "Güte/Untergrenze"}
DEFAULT_DECIDER = "roh_linear"
THETA = 0.1                                   # Untergrenze der Güte des Entscheiders mit Untergrenze
UNDER_FLOOR = -1e9                            # Nutzen einer Alternative unter der Untergrenze

EPS_COMPATIBLE = 1e-7                         # Mindestabstand zwischen aufeinanderfolgenden Referenzen in der Menge der verträglichen Funktionen
TOL_NECESSARY = 1e-9                          # min (u(a) - u(b)) >= -TOL: notwendig

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 12
EXP_M = 5
EXP_MATRICES = 16                             # Matrizen je Experiment (jede Zelle rechnet zwei LPs je Paar)
EXP_SAMPLING_MATRICES = 24
EXP_DRAWS = 100000
EXP_RS = (3, 5, 7, 9)
EXP_RANDOM_VERTICES = 6
OFFSET_REFS = 104729                          # Strom der Referenzauswahl: SplitMix64(seed + OFFSET_REFS + r)
OFFSET_NOISE = 7919                           # Strom des Urteilsrauschens
OFFSET_VERTEX = 15485863                      # Strom der zufälligen verträglichen Ecken
OFFSET_DRAWS = 32452843                       # Strom der Gewichtsziehungen im Kontrast zu SMAA

COLOR_NEC = "#2F6B65"
COLOR_UNDEC = "#e8e2d4"
COLOR_WRONG = "#c0392b"
COLOR_LINE = "rgba(80,80,80,0.7)"
COLOR_CRIT = ("#4c78a8", "#D68A2E", "#2F6B65", "#7a5195", "#8A96A6", "#c0392b", "#59a14f", "#b07aa1")

PRESETS = {
    "Standardfall (6 Referenzen, nur Gewichte)": {"n": DEFAULT_N, "m": DEFAULT_M, "r": 6, "gamma": 1, "sigma": 0.0, "decider": "roh_linear", "seed": DEFAULT_SEED},
    "Wenige Beispiele": {"n": 12, "m": 5, "r": 2, "gamma": 1, "sigma": 0.0, "decider": "roh_linear", "seed": 5},
    "Flexible Randfunktionen": {"n": 12, "m": 5, "r": 6, "gamma": 3, "sigma": 0.0, "decider": "roh_linear", "seed": 5},
    "Nicht additiv": {"n": 12, "m": 5, "r": 6, "gamma": 1, "sigma": 0.0, "decider": "untergrenze", "seed": 32},
    "Verrauschte Urteile": {"n": 12, "m": 5, "r": 6, "gamma": 1, "sigma": 0.2, "decider": "roh_linear", "seed": 5},
}
PRESET_HELP = {
    "Standardfall (6 Referenzen, nur Gewichte)": "12 Alternativen, der Entscheider ordnet 6; ein Teil der übrigen Paare ist notwendig geordnet, alle richtig. Der Rest bleibt offen.",
    "Wenige Beispiele": "Nur 2 Referenzen: fast nichts ist notwendig, die Rangintervalle sind breit. Eine einzelne Lösung wäre reine Wahl.",
    "Flexible Randfunktionen": "Stückweise lineare Randfunktionen (3 Abschnitte): dieselben Beispiele entscheiden deutlich weniger Paare als bei nur Gewichten.",
    "Nicht additiv": "Entscheider mit Untergrenze: sein Nutzen ist nicht additiv; die Ordnung der Referenzen ist nicht verträglich oder ein „notwendiges“ Paar ist falsch.",
    "Verrauschte Urteile": "Die Urteile sind um 20 % verrauscht: die Ordnung der Referenzen widerspricht dem Modell, es gibt keine verträgliche Funktion.",
}
