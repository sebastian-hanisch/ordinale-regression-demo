"""Aus Beispielen lernen: UTA und robuste ordinale Regression - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwölftes Stück der Linie Mehrkriterien-Ranking (Kind von „Gewichte erheben“, Kontrast zu SMAA). Dieselbe neutrale Entscheidungsmatrix; ein simulierter Entscheider ordnet einige
Referenzalternativen. UTA findet eine additive Wertfunktion per LP, die robuste ordinale Regression alle verträglichen: notwendig und möglich besser.
Kern: ord_uta.py, ord_judgments.py, ord_score.py; Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ord_constants as C
from ord_evaluation import (
    Settings,
    analyse,
    de,
    experiment_arbitrary,
    experiment_examples,
    experiment_model,
    experiment_sampling,
    verdict_arbitrary,
    verdict_examples,
    verdict_model,
    verdict_sampling,
)
from ord_pdf_export import build_pdf
from ord_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ord_visualization import (
    alternatives_table,
    arbitrary_table,
    build_examples,
    build_intervals,
    build_marginals,
    build_relations,
    examples_table,
    model_table,
    reference_table,
    sampling_table,
)

st.set_page_config(page_title="Aus Beispielen lernen: ordinale Regression – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Experiment 1 rechnet …")
def _exp_examples():
    return experiment_examples()


@st.cache_data(show_spinner="Experiment 2 rechnet …")
def _exp_model():
    return experiment_model()


@st.cache_data(show_spinner="Experiment 3 rechnet …")
def _exp_arbitrary():
    return experiment_arbitrary()


@st.cache_data(show_spinner="Experiment 4 rechnet …")
def _exp_sampling():
    return experiment_sampling()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🧭 Aus Beispielen lernen: UTA und robuste ordinale Regression")
st.markdown(
    """
**Aus einigen geordneten Beispielen wird nur dann „notwendig“, was für alle verträglichen Wertfunktionen gilt, und das braucht mehr Beispiele, als man denkt.** Dies ist das zwölfte Stück
der Linie **Mehrkriterien-Ranking**: Der simulierte Entscheider ordnet einige **Referenzalternativen**; **UTA** findet eine additive Wertfunktion per LP, die **robuste ordinale Regression**
alle verträglichen Funktionen und damit, für welche Paare gilt: **notwendig** besser (für alle) oder nur **möglich** besser (für mindestens eine). Die Demo misst, wie viele Beispiele es
braucht, was passiert, wenn das Modell nicht passt, wie willkürlich eine einzelne Lösung ist und was die Ziehung gegen den LP kostet (Kontrast zu SMAA).

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die ordinale Regression", expanded=True):
    st.markdown(
        """
1. **Beispielurteile:** Der Entscheider ordnet r Referenzen vollständig. Die übrigen Alternativen kennt er auch, aber er urteilt nicht über sie; sie werden bewertet.
2. **Wertfunktion:** u(a) = Summe der Randfunktionen je Kriterium; jede ist nichtfallend und stückweise linear mit γ gleich langen Abschnitten (γ = 1: nur Gewichte). u(1, …, 1) = 1.
3. **UTA:** per LP die Funktion, die die Ordnung der Referenzen reproduziert und den kleinsten Abstand zwischen aufeinanderfolgenden maximiert (ε). Ist ε ≤ 0, widerspricht die Ordnung der
   Modellklasse.
4. **Robust:** *notwendig* besser: u(a) ≥ u(b) für **alle** verträglichen Funktionen (kleinster Wert von u(a) − u(b) mindestens 0); *möglich* besser: für **mindestens eine**. Je geordnetes
   Paar ein LP. Das **Rangintervall** einer Alternative reicht vom besten Rang (1 + Zahl der notwendig besseren) bis zum schlechtesten.
5. **Eine Lösung ist eine Wahl:** Die Lösung mit größtem Abstand ist eine von vielen verträglichen; auf den offenen Paaren hängt ihre Ordnung an dieser Wahl.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen es gibt; zwei davon bleiben mindestens zum Bewerten übrig.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist.")
    st.markdown("**Beispielurteile**")
    r = st.slider("Geordnete Referenzen r", *bounds("r_slider"), key="r_slider", help="Wie viele Alternativen der Entscheider ordnet (höchstens n − 2 wirken).")
    sigma = st.selectbox("Rauschen der Urteile", C.SIGMA_CHOICES, key="sigma_select", format_func=lambda v: f"{de(100 * v, 0)} %",
                         help="Wie stark der Entscheider von seinem eigenen Nutzen abweicht: Nutzen mal exp(σ Z). 0 % = widerspruchsfrei.")
    st.markdown("**Modell**")
    gamma = st.selectbox("Randfunktionen", C.GAMMA_CHOICES, key="gamma_select", format_func=lambda x: C.GAMMA_LABELS[x],
                         help="Wie flexibel die Randfunktionen sind: je flexibler, desto weniger Paare sind notwendig entschieden.")
    st.markdown("**Entscheider (simuliert)**")
    decider = st.selectbox("Entscheider", C.DECIDERS, key="decider_select", format_func=lambda x: C.DECIDER_LABELS[x],
                           help="Wie der verborgene Entscheider bewertet. Nur der mit Rohwerten liegt in der Modellklasse bei γ = 1.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "r_slider": int(r), "gamma_select": int(gamma), "sigma_select": float(sigma), "decider_select": decider,
                   "seed_input": int(seed)})

settings = Settings(int(n), int(m), int(r), int(gamma), float(sigma), decider, int(seed))
a = _analysis(settings)
c, met = a.case, a.metrics

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Referenzen geordnet", f"{a.r_eff} von {settings.n}", delta=f"{met['k']} Alternativen bleiben, {len(c.pairs)} Urteilspaare", delta_color="off",
          help="Wie viele Alternativen der Entscheider geordnet hat und wie viele aufeinanderfolgende Paare daraus entstehen.")
c2.metric("Ordnung verträglich", "ja" if c.compatible else "nein", delta=f"größter Abstand ε = {de(c.eps, 3)}", delta_color="off",
          help="Ob es eine additive Wertfunktion dieser Klasse gibt, die die Ordnung der Referenzen streng reproduziert (ε > 0).")
c3.metric("Paare notwendig entschieden", "–" if met["decided"] is None else f"{de(100 * met['decided'], 1)} %",
          delta="keine verträgliche Funktion" if met["decided"] is None else (f"davon richtig {de(100 * met['correct'], 1)} %" if met["correct"] is not None else "kein Paar entschieden"),
          delta_color="off", help="Anteil der Paare unter den übrigen Alternativen, für die u(a) ≥ u(b) für alle verträglichen Funktionen gilt (Ordnung des Entscheiders eindeutig).")
c4.metric("Mittlere Rangbreite", "–" if met["width"] is None else de(met["width"], 1),
          delta=None if met["possible_best"] is None else f"Rang 1 möglich für {met['possible_best']} Alternativen", delta_color="off",
          help="Mittlere Breite des Rangintervalls (schlechtester minus bester Rang) unter den übrigen Alternativen.")

left, right_col = st.columns([3, 2])
with left:
    rel = build_relations(a)
    if rel is not None:
        st.markdown("**Relationen unter den übrigen Alternativen** (≻ Zeile notwendig besser, ≺ notwendig schlechter, ? offen)")
        st.plotly_chart(rel, width="stretch", key="relations_chart")
    else:
        st.warning("Die Ordnung der Referenzen ist mit keiner Funktion der Klasse verträglich: Es gibt keine verträglichen Funktionen, also keine notwendigen oder möglichen Aussagen.")
with right_col:
    st.markdown("**Beispielurteile** (Ordnung des Entscheiders)")
    st.dataframe(reference_table(a), hide_index=True, width="stretch")

iv = build_intervals(a)
if iv is not None:
    st.markdown("**Rangintervalle** der übrigen Alternativen: Balken = von bester bis schlechtester möglicher Rang")
    st.plotly_chart(iv, width="stretch", key="intervals_chart")
    st.caption("Raute: Rang nach dem wahren Nutzen des Entscheiders; Kreuz: Rang der Lösung mit größtem Abstand (eine von vielen verträglichen).")

tab_t, tab_m = st.tabs(["Alle übrigen Alternativen", "Randfunktionen der Lösung"])
with tab_t:
    st.dataframe(alternatives_table(a), hide_index=True, width="stretch")
with tab_m:
    st.plotly_chart(build_marginals(a), width="stretch", key="marginals_chart")
    st.caption("Beitrag jedes Kriteriums zum Nutzen der Lösung mit größtem Abstand; der Endwert ist sein Gewicht. Eine andere verträgliche Lösung sieht anders aus.")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="ordinale_regression.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann wird aus möglich notwendig?")
if not c.compatible:
    st.warning("Die Ordnung der Referenzen widerspricht der Modellklasse (kein ε > 0): Entweder passt die Klasse nicht (nicht additiver Entscheider) oder die Urteile sind verrauscht. "
               "Die angezeigte Lösung ist die mit der kleinsten Verletzung; „notwendig“ lässt sich nicht aussagen.")
elif met["wrong_n"]:
    st.warning(f"Von den notwendig entschiedenen Paaren sind {met['wrong_n']} falsch: Das Modell passt nicht zum Entscheider, die Voraussetzung von „notwendig“ ist verletzt.")
elif met["decided"] is not None:
    open_pairs = round(met["n_pairs"] * (1 - met["decided"]))
    st.info(f"Alle notwendig entschiedenen Paare stimmen mit dem Entscheider überein; {open_pairs} von {met['n_pairs']} Paaren bleiben offen. Auf den offenen Paaren hängt jede einzelne "
            f"Lösung an der Wahl: Die mit größtem Abstand trifft dort {'–' if met['undecided_acc'] is None else de(100 * met['undecided_acc'], 0) + ' %'}.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption(f"Feste Matrizen (SWEEP_SEEDS), {C.EXP_N} Alternativen, {C.EXP_M} Kriterien, unabhängig von den Reglern; jedes Experiment rechnet rund 20 Sekunden (zwei LPs je Paar) und startet "
           "erst auf Klick. Wenige Matrizen: Unterschiede unter etwa 10 Punkten sind nicht belastbar.")

st.markdown("### Experiment 1: Wie viele Beispielurteile?")
st.caption(f"Entscheider mit Rohwerten (in der Modellklasse), widerspruchsfreie Urteile; {C.EXP_MATRICES} Matrizen. Anteil der Paare unter den übrigen Alternativen, die notwendig entschieden sind.")
if st.button("▶ Experiment 1 rechnen", key="exp_examples_start"):
    st.session_state["exp_examples"] = _exp_examples()
if st.session_state.get("exp_examples"):
    rows = st.session_state["exp_examples"]
    st.plotly_chart(build_examples(rows), width="stretch", key="examples_chart")
    verdict(verdict_examples(rows))
    st.dataframe(examples_table(rows), hide_index=True, width="stretch")

st.markdown("### Experiment 2: Passt das Modell, sind die Urteile widerspruchsfrei?")
st.caption(f"6 Referenzen; je Entscheider ohne Rauschen und der Entscheider mit Rohwerten mit Rauschen; {C.EXP_MATRICES} Matrizen. Verträglich heißt: eine Funktion der Klasse reproduziert die Ordnung.")
if st.button("▶ Experiment 2 rechnen", key="exp_model_start"):
    st.session_state["exp_model"] = _exp_model()
if st.session_state.get("exp_model"):
    rows = st.session_state["exp_model"]
    verdict(verdict_model(rows))
    st.dataframe(model_table(rows), hide_index=True, width="stretch")

st.markdown("### Experiment 3: Wie willkürlich ist eine einzelne Lösung?")
st.caption("Auf den nicht entschiedenen Paaren: die Lösung mit größtem Abstand gegen zufällige verträgliche Ecken des Raums der Funktionen; wie oft zwei zufällige Ecken verschieden ordnen.")
if st.button("▶ Experiment 3 rechnen", key="exp_arbitrary_start"):
    st.session_state["exp_arbitrary"] = _exp_arbitrary()
if st.session_state.get("exp_arbitrary"):
    rows = st.session_state["exp_arbitrary"]
    verdict(verdict_arbitrary(rows))
    st.dataframe(arbitrary_table(rows), hide_index=True, width="stretch")

st.markdown("### Experiment 4: Gesampelt gegen ausgelotet")
draws_text = f"{C.EXP_DRAWS:,}".replace(",", " ")
st.caption(f"Nur Gewichte (γ = 1): Anteil der {draws_text} gleichverteilten Gewichtsziehungen, die die Ordnung der Referenzen erfüllen, und mögliche Beste in der Stichprobe gegen den LP "
           f"({C.EXP_SAMPLING_MATRICES} Matrizen).")
if st.button("▶ Experiment 4 rechnen", key="exp_sampling_start"):
    st.session_state["exp_sampling"] = _exp_sampling()
if st.session_state.get("exp_sampling"):
    rows = st.session_state["exp_sampling"]
    verdict(verdict_sampling(rows))
    st.dataframe(sampling_table(rows), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Modell ist additiv und stückweise linear** | Ein nicht additiver Entscheider (Untergrenze) erzeugt falsche „notwendige“ Paare oder eine unverträgliche Ordnung (Experiment 2). | Kriterien-Wechselwirkung |
| **Die Urteile sind widerspruchsfrei** | Mit Rauschen ist die Ordnung oft nicht verträglich, und entschiedene Paare sind falsch (Experiment 2). | Gruppen-Entscheidung |
| **Flexibilität kostet Entscheidbarkeit** | Mehr Abschnitte je Randfunktion lassen mehr Funktionen zu, also weniger notwendige Paare (Experiment 1). | – |
| **Die Referenzen sind zufällig gewählt** | Informative Referenzen (an den Rändern) würden schneller entscheiden; die Demo misst den Durchschnitt. | – |
| **Keine Intensitäten (GRIP), keine Gleichstände** | Urteile der Art „um wie viel besser“ und gleich gute Referenzen sind nicht Teil. | – |
| **Der Entscheider ist ein Orakel** | Kein Mensch, der zögert oder seine Meinung ändert. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Wertfunktion.** $u(a) = \sum_{j=1}^{m} u_j(x_j(a))$ mit $x_j(a) \in [0, 1]$ (Nenn-Spannweite) und $u_j$ nichtfallend, stückweise linear auf $\gamma$ gleich langen Abschnitten:
$u_j(x) = \sum_{k=1}^{\gamma} d_{jk} \cdot \min(1, \max(0, \gamma x - (k - 1)))$ mit $d_{jk} \ge 0$ und $\sum_{j,k} d_{jk} = 1$. Für $\gamma = 1$ sind $d_j$ die Gewichte.

**UTA.** Für aufeinanderfolgende Referenzen $a \succ b$: $u(a) - u(b) \ge \varepsilon$; maximiere $\varepsilon$. Die Ordnung ist verträglich, wenn $\varepsilon^\ast > 0$.

**Notwendig und möglich.** Mit der Menge $U^C$ der verträglichen Funktionen: $a \succeq^N b \iff \min_{u \in U^C} (u(a) - u(b)) \ge 0$ und $a \succeq^P b \iff \max_{u \in U^C} (u(a) - u(b)) \ge 0$
(je ein lineares Programm). *Strikt notwendig:* $a \succeq^N b$ und nicht $b \succeq^N a$.

**Rangintervall.** Bester Rang $1 + |\{b : b \succ^N a\}|$, schlechtester Rang $k - |\{b : a \succ^N b\}|$ unter den $k$ übrigen Alternativen.

**Gesampelt gegen ausgelotet ($\gamma = 1$).** Der Raum $\{w \in \Delta : (x(a) - x(b))^\top w \ge 0\ \forall (a, b)\}$ ist ein Polytop; seine Menge möglicher Bester ergibt ein LP je Alternative,
Ziehungen aus $\Delta$ treffen ihn mit einem Anteil, der mit jedem Urteil fällt.

**Literatur:** UTA, UTAGMS und GRIP sind hier nicht im Original geprüft und deshalb nicht angegeben.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html)."
)
