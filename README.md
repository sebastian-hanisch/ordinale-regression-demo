# Aus Beispielen lernen: UTA und robuste ordinale Regression – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-ordinale-regression-demo.streamlit.app/)**

Zwölftes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind von
[Gewichte erheben](https://github.com/sebastian-hanisch/gewichte-erheben-demo) und Kontrast zu [SMAA](https://github.com/sebastian-hanisch/smaa-demo). Der simulierte Entscheider ordnet
einige **Referenzalternativen**. **UTA** findet per LP eine additive Wertfunktion, die diese Ordnung reproduziert; die **robuste ordinale Regression** betrachtet alle verträglichen
Funktionen und sagt, welche Paare **notwendig** (für alle) und welche nur **möglich** (für mindestens eine) besser sind.

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Der **Entscheider** kennt seinen
Nutzen (linear, konkav oder Cobb-Douglas in der Güte, linear in den Rohwerten, oder mit Untergrenze) und urteilt nur über die Referenzen. Er ist ein Orakel, kein Mensch.

## Warum dieses Problem

**Aus einigen geordneten Beispielen wird nur dann „notwendig“, was für alle verträglichen Wertfunktionen gilt, und das braucht mehr Beispiele, als man denkt.** Die Kernfrage der Demo:
Wie viele Beispielurteile, bis aus möglich notwendig wird, und wann ist „notwendig“ falsch?

## Befunde und Korrekturen gegenüber dem Plan

- **Bei widerspruchsfreien Urteilen ist notwendig in der Modellklasse richtig, aber teuer.** Bei einem Entscheider mit Rohwerten sind 100,0 % der entschiedenen Paare richtig; bei 3 geordneten Referenzen von 12
  Alternativen sind 13,2 % der übrigen Paare entschieden, bei 9 sind es 54,2 % (aus nur 16 Matrizen; in 200 Matrizen rund 65 %). Mit stückweise linearen Randfunktionen (3 Abschnitte): 6,2 % und 14,6 %.
- **Das Modell muss passen.** Beim nicht additiven Entscheider (Untergrenze) sind bei nur-Gewichten 5,7 % der „notwendigen“ Paare falsch (0,38 je Matrix); mit 3 Abschnitten verschwindet
  das, aber nur 8,6 % der Paare sind dann entschieden.
- **Die Urteile müssen widerspruchsfrei sein.** Mit 10 % Rauschen ist die Ordnung nur in 93,8 % der Matrizen verträglich und 4,4 % der entschiedenen Paare (unter den verträglichen Matrizen) sind falsch; mit 20 % in
  68,8 % und 6,1 %.
- **Eine einzelne Lösung ist eine Wahl:** Bei 6 Referenzen bleiben 63,8 % der Paare offen. Dort ordnet die Lösung mit größtem Abstand 78,5 % richtig, eine zufällige verträgliche Ecke
  67,9 %, und zwei Ecken ordnen 40,8 % dieser Paare verschieden.
- **Ohne Urteile ist nur die Dominanz notwendig** (Pareto): ein Zusammenhang mit der Wurzel der Linie, den der Test `test_no_reference_and_one_reference_give_no_judgment` festhält.
- **Gesampelt gegen ausgelotet:** Von 100 000 Gewichtsziehungen erfüllen bei 3 Referenzen im Mittel gut ein Drittel die Ordnung (36,8 % in den 24 Matrizen, die Streuung zwischen Matrizen ist groß), bei 8 nur rund 1,5 %; der LP kostet je Alternative eine Rechnung.
- **Jedes Experiment startet auf eigenen Klick** (rund 20 Sekunden: zwei LPs je Paar).

## Modell

- **Matrix:** n Alternativen (8 bis 20, Standard 12), m Kriterien (3 bis 8), Rohwerte mit Richtung und Einheit, auf die feste Nenn-Spannweite normiert.
- **Beispielurteile:** r zufällig gewählte Referenzen (2 bis 12; höchstens n − 2 wirken), vollständig nach dem Nutzen des Entscheiders geordnet, optional mit Rauschen
  (Nutzen · exp(σ Z), σ = 0,1 / 0,2). Die übrigen Alternativen werden bewertet.
- **Wertfunktion:** u(a) = Σⱼ uⱼ(xⱼ(a)), uⱼ nichtfallend und stückweise linear mit γ = 1 (nur Gewichte) oder 3 gleich langen Abschnitten; u(1, …, 1) = 1.
- **UTA:** per LP der größte Abstand ε zwischen aufeinanderfolgenden Referenzen. **Robust:** für jedes geordnete Paar min und max von u(a) − u(b) über die verträglichen Funktionen.
- **Fester Rahmen der Experimente:** feste Matrizen (`SWEEP_SEEDS`), 12 Alternativen, 5 Kriterien, 16 Matrizen je Experiment (24 beim Sampling-Kontrast), unabhängig von den Reglern.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselbe Matrix, derselbe Entscheider:** Die Lösung und die Relationen werden an den wahren Nutzen des Entscheiders gehalten, nicht an eine Formel.
2. **Messgröße ist die Zuordnung der Paare:** entschieden, davon richtig, Rangbreite, mögliche Beste, Verträglichkeit (ε > 0).
3. **Gegenproben:** Modellklasse (Rohwerte gegen Güte gegen Untergrenze), Flexibilität (γ), Rauschen.
4. **Kontrast zu SMAA:** dieselbe Menge verträglicher Gewichte, gezogen gegen ausgelotet.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: `tests/test_claims.py`. 16 feste Matrizen je Experiment (24 beim Sampling), 12 Alternativen, 5 Kriterien. **Der Standardfehler eines Anteils ist 2 bis 9 Punkte;
Unterschiede unter etwa 10 Punkten sind nicht belastbar und stehen nicht als Aussage; belastbar sind die Richtung der Verläufe und die Identitäten (100,0 %).**

| Frage | Befund | Test |
|---|---|---|
| Wie viele Beispielurteile? Entscheider mit Rohwerten, r = 3 / 5 / 7 / 9 von 12: Paare entschieden (nur Gewichte; 3 Abschnitte) | **13,2 / 30,4 / 51,2 / 54,2 %; 6,2 / 9,5 / 13,1 / 14,6 %**; davon richtig 100,0 %; Rangbreite 6,9 / 4,2 / 2,0 / 0,9 (3 Abschnitte 7,5 / 5,4 / 3,5 / 1,7); Rang 1 möglich für 6,25 / 3,88 / 2,50 / 1,81 Alternativen; notwendig beste 0,0 / 0,19 / 0,31 / 0,38. | `test_how_many_judgments_until_possible_becomes_necessary` |
| Modellklasse, 6 Referenzen, nur Gewichte: Ordnung verträglich / entschieden / richtig / falsche notwendige Paare je Matrix | Rohwerte/linear **100 % / 36,3 % / 100,0 % / 0,00**; Güte/linear 100 % / 38,7 % / 98,4 % / 0,06; **Untergrenze 100 % / 34,8 % / 94,3 % / 0,38**; mit 3 Abschnitten Untergrenze richtig 100,0 %. | `test_the_model_class_and_the_noise_decide_whether_necessary_is_right` |
| Rauschen der Urteile (Rohwerte/linear, 6 Referenzen, nur Gewichte): verträglich / richtig / falsche Paare je Matrix | σ = 0,1: **93,8 % / 95,6 % / 0,40**; σ = 0,2: **68,8 % / 93,9 % / 0,55**; mit 3 Abschnitten verträglich 100 % und 87,5 %. | dasselbe |
| Eine einzelne Lösung (nur Gewichte; r = 4 / 6 / 8): offene Paare / Lösung mit größtem Abstand richtig / zufällige Ecke richtig / zwei Ecken verschieden | **79,0 / 63,8 / 49,0 %; 82,8 / 78,5 / 73,2 %; 62,3 / 67,9 / 58,9 %; 40,0 / 40,8 / 44,7 %**; mit 3 Abschnitten offen 93,5 / 90,4 / 84,4 %, verschieden 31,8 / 32,2 / 35,7 %. | `test_a_single_solution_is_a_choice` |
| Gesampelt gegen ausgelotet (nur Gewichte; r = 3 / 4 / 6 / 8; 100 000 Ziehungen): Anteil verträglich / mögliche Beste per LP / in der Stichprobe | **36,8 / 19,6 / 4,9 / 1,5 %; 5,58 / 4,17 / 2,50 / 1,92; 5,21 / 3,71 / 2,21 / 1,75**; der Beste des Entscheiders ist in jeder der 24 Matrizen im LP-Raum möglich. | `test_sampling_hits_the_compatible_region_ever_more_rarely` |

## Ehrliche Grenzen

- **Notwendig gilt nur unter Voraussetzungen.** Modellklasse (additiv, stückweise linear) und widerspruchsfreie Urteile; die Experimente zeigen falsche notwendige Paare (bis 5,7 %)
  und unverträgliche Ordnungen (bis 31 % der Matrizen).
- **Flexibilität kostet.** Mit 3 Abschnitten sind bei 7 Referenzen 13,1 statt 51,2 % der Paare entschieden.
- **Die Referenzen sind zufällig gewählt.** Ob eine gezielte Auswahl schneller entscheidet, misst die Demo nicht; sie misst den Durchschnitt zufälliger Referenzen.
- **Eine Lösung ist keine Antwort.** Die Lösung mit größtem Abstand ist eine von vielen verträglichen.
- **Kein GRIP, keine Gleichstände, keine Intensitäten:** Urteile der Art „um wie viel besser“ sind nicht Teil; UTA und seine Weiterentwicklungen sind hier nicht im Original geprüft.
- **Wenige Matrizen.** 16 je Experiment wegen der Rechenzeit (zwei LPs je Paar); Unterschiede unter etwa 10 Punkten sind Rauschen.

## Verifikation

- **Kern:** Basisfunktionen, UTA (maximin), Notwendig und Möglich, Rangintervall, zufällige Ecken und der Gewichtsraum gegen die Handrechnung der Mini-Instanz (`tests/conftest.py`: m = 2,
  Referenzurteil c vor b, w ≤ 3/7; notwendig b ≻ a, b ≻ d, c ≻ a, c ≻ b, c ≻ d; offen a gegen d).
- **Orakel** (`tests/test_oracle_ordinal.py`): in der Modellklasse nie ein falsches notwendiges Paar (γ = 1, 2, 3; Gegenprobe mit nicht additivem Nutzen); Antisymmetrie und
  notwendig ⇒ möglich; Monotonie in den Beispielen; m = 2 gegen ein feines Gitter; LP gegen Stichprobe.
- **Randfälle** (`tests/test_edge_cases.py`): r = 0 und 1 (nur Dominanz notwendig), r größer als n − 2, Gleichstand der Referenzen, unverträgliche Ordnung, m = 1, gleiche Alternativen.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`; Verdikte an von Hand gesetzten Zeilen in `test_verdicts.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Mutationstest** vor dem ersten Commit (`tools/mutate_demo.py` der Website); Stand in `tests/mutation_state.json`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, jede Auswahl), alle Presets, unverträgliche Ordnung, Permalink, PDF und jedes Experiment erst nach
  eigenem Klick, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `ord_constants.py` | Regler-Grenzen, Entscheider, Randfunktionen, Rauschen, feste Annahmen der Experimente, Presets |
| `ord_score.py` | Nenn-Spannweite, Nutzen der Entscheider, Kendalls τ-b |
| `ord_judgments.py` | Referenzen wählen, ordnen (mit Rauschen), Paare aufeinanderfolgender Referenzen |
| `ord_uta.py` | Basisfunktionen, UTA (maximin), notwendig und möglich per LP, Rangintervalle, zufällige Ecken, Gewichtsraum gegen Stichprobe |
| `ord_evaluation.py` | Fall der App, vier Experimente, Verdikte (reine Funktionen) |
| `ord_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `ord_visualization.py` | Relationsmatrix, Rangintervalle, Randfunktionen, Tabellen und Abbildungen der Experimente (Achsen gesperrt) |
| `ord_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanz, UTA, Urteile, Score, Orakel, Randfälle, Generator, Auswertung, Verdikte, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **GRIP und Intensitätsurteile („um wie viel besser“):** nur die Ordnung der Referenzen.
- **Intervalle für die Randfunktionen und Mehrfach-Urteile über Paare nicht aufeinanderfolgender Referenzen:** nur Paare aufeinanderfolgender Referenzen.
- **Die repräsentative Lösung der robusten Regression:** nur die Lösung mit größtem Abstand und zufällige Ecken.
- **Informative Auswahl der Referenzen (aktives Lernen):** die Referenzen sind zufällig.
- **Mehrere Entscheider und Konflikt der Urteile:** steht in einem eigenen Stück der Linie.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html).
