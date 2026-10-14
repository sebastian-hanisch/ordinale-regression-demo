"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import ord_constants as C
from ord_evaluation import Analysis, de

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "↔": "<->", "–": "-", "—": "-", "τ": "tau", "γ": "gamma", "σ": "sigma", "ε": "epsilon", "≥": ">=", "≤": "<=", "≠": "!=", "−": "-",
            "≻": ">", "≺": "<", "€": "EUR", "·": "*", "★": "*"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s, c, m = a.settings, a.case, a.metrics
    names = [f"A{i + 1}" for i in range(s.n)]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Aus Beispielen lernen: UTA und robuste ordinale Regression"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{s.n} Alternativen, {s.m} Kriterien, {a.r_eff} Referenzen geordnet, {s.gamma} Abschnitt(e) je Randfunktion, Rauschen der Urteile {de(100 * s.sigma, 0)} %. "
                            f"Entscheider: {C.DECIDER_LABELS[s.decider]}. Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    if not c.compatible:
        pdf.multi_cell(0, 5, _s(f"Die Ordnung der Referenzen ist mit keiner additiven Wertfunktion dieser Klasse verträglich (epsilon = {de(c.eps, 3)}): es gibt keine notwendigen Aussagen."),
                       new_x="LMARGIN", new_y="NEXT")
    elif m["decided"] is not None:
        pdf.multi_cell(0, 5, _s(f"Von {m['n_pairs']} Paaren unter den {m['k']} übrigen Alternativen sind {de(100 * m['decided'], 1)} % notwendig entschieden"
                                f"{'' if m['correct'] is None else ', davon ' + de(100 * m['correct'], 1) + ' % richtig'}. Mittlere Rangbreite {de(m['width'], 1)}; "
                                f"mögliche Beste: {m['possible_best']}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Nicht-Referenzen (wahrer Rang, Rang der Lösung mit größtem Abstand, Rangintervall)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for idx, i in enumerate(c.others):
        interval = "" if a.best_rank is None else f", Rangintervall {int(a.best_rank[idx])} bis {int(a.worst_rank[idx])}"
        pdf.cell(0, 5, _s(f"{names[i]}: wahrer Rang {int(a.true_rank[idx])}, Rang der Lösung {int(a.uta_rank[idx])}{interval}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Eine einzelne Lösung ist eine von vielen verträglichen. Der Entscheider ist ein simuliertes Orakel, kein Mensch. Quelle: ordinale-regression-demo, "
                            "Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
