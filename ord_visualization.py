"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import ord_constants as C
from ord_evaluation import Analysis, de


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def reference_table(a: Analysis) -> pd.DataFrame:
    """Die Beispielurteile in der Reihenfolge des Entscheiders (bester zuerst); bei Rauschen ist das seine verrauschte Ordnung."""
    names = alt_names(a.settings.n)
    c = a.case
    order = sorted(c.refs, key=lambda i: (-c.observed[i], i))
    return pd.DataFrame({"Platz": list(range(1, len(order) + 1)), "Alternative": [names[i] for i in order],
                         "Nutzen nach dem Urteil": [None if c.observed[i] < -1e8 else float(c.observed[i]) for i in order]})


def build_relations(a: Analysis) -> go.Figure | None:
    """Relationsmatrix der Nicht-Referenzen: Zeile notwendig besser als Spalte (grün), Zeile notwendig schlechter (rot-braun), offen (hell). None, wenn die Ordnung unverträglich ist."""
    c = a.case
    if c.nec is None:
        return None
    names = alt_names(a.settings.n)
    labels = [names[i] for i in c.others]
    k = len(labels)
    z = np.zeros((k, k))
    z[c.nec] = 1.0
    z[c.nec.T] = -1.0
    text = [["" if i == j else ("≻" if z[i, j] > 0 else "≺" if z[i, j] < 0 else "?") for j in range(k)] for i in range(k)]
    scale = [[0.0, "#b9774f"], [0.5, C.COLOR_UNDEC], [1.0, C.COLOR_NEC]]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, text=text, texttemplate="%{text}", colorscale=scale, zmin=-1, zmax=1, showscale=False,
                               hovertemplate="%{y} gegen %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Zeile gegen Spalte")
    fig.update_xaxes(side="top")
    return _fixed(fig, max(300, 34 * k + 100))


def build_intervals(a: Analysis) -> go.Figure | None:
    """Rangintervall je Nicht-Referenz (bester bis schlechtester Rang aus den notwendigen Relationen); Raute: Rang nach dem wahren Nutzen, Kreuz: Rang der Lösung mit größtem Abstand."""
    c = a.case
    if a.best_rank is None:
        return None
    names = alt_names(a.settings.n)
    labels = [names[i] for i in c.others]
    k = len(labels)
    fig = go.Figure()
    for i in range(k):
        fig.add_trace(go.Scatter(x=[a.best_rank[i], a.worst_rank[i]], y=[labels[i]] * 2, mode="lines", line=dict(color=C.COLOR_CRIT[4], width=8), showlegend=False,
                                 hovertemplate=f"{labels[i]}: Rang {a.best_rank[i]} bis {a.worst_rank[i]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.true_rank, y=labels, mode="markers", name="wahrer Rang", marker=dict(symbol="diamond", size=12, color="black")))
    fig.add_trace(go.Scatter(x=a.uta_rank, y=labels, mode="markers", name="Lösung mit größtem Abstand", marker=dict(symbol="x", size=11, color=C.COLOR_CRIT[1])))
    fig.update_xaxes(title="Rang unter den Nicht-Referenzen (1 = beste)", range=[0.5, k + 0.5], dtick=1)
    fig.update_yaxes(autorange="reversed")
    return _fixed(fig, max(300, 30 * k + 120))


def build_marginals(a: Analysis) -> go.Figure:
    """Randfunktionen der Lösung mit größtem Abstand je Kriterium (Wert 0 bis 1 gegen den normierten Kriterienwert); der Endwert ist das Gewicht des Kriteriums."""
    s = a.settings
    g = s.gamma
    xs = np.linspace(0.0, 1.0, g + 1)
    fig = go.Figure()
    for j in range(s.m):
        fig.add_trace(go.Scatter(x=xs, y=a.marginals[j], mode="lines+markers", name=f"K{j + 1}", line=dict(color=C.COLOR_CRIT[j % len(C.COLOR_CRIT)], width=3)))
    fig.update_xaxes(title="normierter Kriterienwert", range=[0, 1])
    fig.update_yaxes(title="Beitrag zum Nutzen", range=[0, 1])
    return _fixed(fig, 320)


def alternatives_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.settings.n)
    c = a.case
    rows = []
    for idx, i in enumerate(c.others):
        row = {"Alternative": names[i], "wahrer Rang": int(a.true_rank[idx]), "Rang der Lösung": int(a.uta_rank[idx])}
        if a.best_rank is not None:
            row["Rangintervall"] = f"{int(a.best_rank[idx])} bis {int(a.worst_rank[idx])}"
            row["notwendig besser als"] = int(c.nec[idx].sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Experimente

def examples_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Referenzen r": f"{r['r']} von {r['n']}", "Randfunktionen": "nur Gewichte" if r["gamma"] == 1 else "3 Abschnitte", "Paare entschieden": f"{de(100 * r['decided'], 1)} %",
                          "davon richtig": f"{de(100 * r['correct'], 1)} %", "Rangbreite": de(r["width"], 1), "mögliche Beste": de(r["possible_best"], 1),
                          "Lösung ordnet richtig": f"{de(100 * r['uta_pairs'], 1)} %"} for r in rows])


def build_examples(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for gamma, label, color in ((1, "nur Gewichte", C.COLOR_CRIT[0]), (3, "3 Abschnitte", C.COLOR_CRIT[1])):
        sel = [r for r in rows if r["gamma"] == gamma]
        fig.add_trace(go.Scatter(x=[r["r"] for r in sel], y=[100 * r["decided"] for r in sel], mode="lines+markers", name=label, line=dict(color=color, width=3)))
    fig.update_xaxes(title="Beispielurteile (geordnete Referenzen)", dtick=1)
    fig.update_yaxes(title="Paare notwendig entschieden (%)", range=[0, 102])
    return _fixed(fig, 340)


def model_table(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        who = C.DECIDER_SHORT[r["decider"]] + ("" if r["kind"] == "decider" else f", Rauschen {de(100 * r['sigma'], 0)} %")
        out.append({"Entscheider": who, "Randfunktionen": "nur Gewichte" if r["gamma"] == 1 else "3 Abschnitte", "Ordnung verträglich": f"{de(100 * r['compatible'], 1)} %",
                    "Paare entschieden": "-" if r["decided"] is None else f"{de(100 * r['decided'], 1)} %",
                    "davon richtig": "-" if r["correct"] is None else f"{de(100 * r['correct'], 1)} %", "falsche notwendige Paare je Matrix": "-" if r["wrong_n"] is None else de(r["wrong_n"], 2)})
    return pd.DataFrame(out)


def arbitrary_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Referenzen r": r["r"], "Randfunktionen": "nur Gewichte" if r["gamma"] == 1 else "3 Abschnitte", "Paare offen": f"{de(100 * r['undecided'], 1)} %",
                          "Lösung mit größtem Abstand richtig": f"{de(100 * r['maximin'], 1)} %", "zufällige Ecke richtig": f"{de(100 * r['random'], 1)} %",
                          "zwei Ecken ordnen verschieden": f"{de(100 * r['disagree'], 1)} %"} for r in rows])


def sampling_table(rows: list[dict]) -> pd.DataFrame:
    def pct(x):
        return de(100 * x, 2) if x < 0.1 else de(100 * x, 1)

    return pd.DataFrame([{"Referenzen r": r["r"], "Ziehungen, die die Ordnung erfüllen": f"{pct(r['share'])} %", "davon (Mittel)": f"{de(r['accepted'], 0)} von {de(r['draws'], 0)}",
                          "mögliche Beste per LP": de(r["possible_lp"], 2), "in der Stichprobe": "-" if r["found"] is None else de(r["found"], 2),
                          "übersehen": "-" if r["missed"] is None else de(r["missed"], 2), "Matrizen ohne Treffer": r["none"]} for r in rows])
