"""Verdikte an von Hand gesetzten Zeilen (DEMO-PLAYBOOK §12): jede Zahl im Text und die Auswahl der Zeilen steht hier."""

import ord_evaluation as ev


def _ex(r, gamma, decided, correct=1.0, width=3.0, n=12):
    return {"r": r, "gamma": gamma, "n": n, "compatible": 1.0, "decided": decided, "correct": correct, "width": width, "possible_best": 3.0, "necessary_best": 0.0,
            "uta_pairs": 0.8, "best_possible": 1.0, "k": n - r}


def test_verdict_examples_names_the_smallest_and_the_largest_number_of_judgments():
    rows = [_ex(3, 1, 0.132, width=6.94), _ex(9, 1, 0.542, width=0.92), _ex(3, 3, 0.062), _ex(9, 3, 0.146), _ex(6, 1, 0.4), _ex(6, 3, 0.1)]
    kind, text = ev.verdict_examples(rows)
    assert kind == "info"
    assert "alle entschiedenen Paare richtig (100,0 %)" in text
    assert "bei 3 Beispielurteilen von 12 Alternativen 13,2 % der übrigen Paare, bei 9 sind es 54,2 %" in text
    assert "mit stückweise linearen Randfunktionen 6,2 % und 14,6 %" in text
    assert "Die Rangbreite sinkt von 6,9 auf 0,9 Plätze" in text


def _mod(kind, decider, sigma, gamma, compatible, correct):
    return {"kind": kind, "decider": decider, "sigma": sigma, "gamma": gamma, "r": 6, "compatible": compatible, "decided": 0.4, "correct": correct, "wrong_n": 0.0, "uta_pairs": 0.8}


def test_verdict_model_picks_the_matched_the_non_additive_and_the_noisy_rows():
    rows = [_mod("decider", "roh_linear", 0.0, 1, 1.0, 1.0), _mod("decider", "untergrenze", 0.0, 1, 1.0, 0.943), _mod("noise", "roh_linear", 0.1, 1, 0.9, 0.95),
            _mod("noise", "roh_linear", 0.2, 1, 0.688, 0.939), _mod("noise", "roh_linear", 0.2, 3, 0.9, 0.99), _mod("decider", "untergrenze", 0.0, 3, 1.0, 1.0)]
    kind, text = ev.verdict_model(rows)
    assert kind == "warning"
    assert "sind 100,0 % der entschiedenen Paare richtig (Ordnung verträglich in 100,0 % der Matrizen)" in text
    assert "Entscheider (Untergrenze) sind 5,7 % der „notwendigen“ Paare falsch" in text
    assert "nur in 68,8 % der Matrizen verträglich, und 6,1 % der entschiedenen Paare (unter den verträglichen Matrizen) sind falsch" in text


def test_verdict_arbitrary_names_open_pairs_and_both_solutions():
    rows = [{"r": 6, "gamma": 1, "undecided": 0.638, "maximin": 0.785, "random": 0.679, "disagree": 0.408},
            {"r": 6, "gamma": 3, "undecided": 0.904, "maximin": 0.777, "random": 0.705, "disagree": 0.322},
            {"r": 4, "gamma": 1, "undecided": 0.79, "maximin": 0.8, "random": 0.6, "disagree": 0.4}]
    kind, text = ev.verdict_arbitrary(rows)
    assert kind == "info"
    assert "Bei 6 Beispielurteilen bleiben 63,8 % der Paare offen" in text
    assert "ordnet die Lösung mit größtem Abstand 78,5 % richtig, eine zufällige verträgliche Ecke 67,9 %" in text
    assert "ordnen 40,8 % der offenen Paare verschieden (mit stückweise linearen Randfunktionen 32,2 %)" in text


def test_verdict_sampling_uses_the_smallest_and_largest_number_of_references():
    rows = [{"r": 3, "share": 0.368, "accepted": 36776.7, "none": 0, "possible_lp": 5.583, "found": 5.208, "missed": 0.375, "in_region": 24, "matrices": 24, "draws": 100000},
            {"r": 8, "share": 0.015, "accepted": 1494.5, "none": 0, "possible_lp": 1.917, "found": 1.75, "missed": 0.167, "in_region": 24, "matrices": 24, "draws": 100000},
            {"r": 6, "share": 0.049, "accepted": 4890.0, "none": 0, "possible_lp": 2.5, "found": 2.2, "missed": 0.3, "in_region": 24, "matrices": 24, "draws": 100000}]
    kind, text = ev.verdict_sampling(rows)
    assert kind == "info"
    assert "Von 100000 gleichverteilten Gewichtsziehungen erfüllen bei 3 Referenzen 36,8 % die Ordnung, bei 8 nur 1,5 %" in text
    assert "im Mittel 1494 Ziehungen" in text or "im Mittel 1495 Ziehungen" in text
    assert "Der LP findet 1,92 mögliche Beste je Matrix, die Stichprobe 1,75; bei 3 sind es 5,58 gegen 5,21" in text
