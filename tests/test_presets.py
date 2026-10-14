"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import ord_constants as C
import ord_evaluation as ev
import ord_presets as P

STANDARD = "Standardfall (6 Referenzen, nur Gewichte)"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_and_choices():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and C.R_MIN <= p["r"] <= C.R_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["gamma"] in C.GAMMA_CHOICES and p["sigma"] in C.SIGMA_CHOICES and p["decider"] in C.DECIDERS, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[STANDARD]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.DECIDER_LABELS) == set(C.DECIDERS) == set(C.DECIDER_SHORT)
    assert set(C.GAMMA_LABELS) == set(C.GAMMA_CHOICES)


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis(STANDARD)
    assert a.case.compatible and 0.25 <= a.metrics["decided"] <= 0.5 and a.metrics["wrong_n"] == 0

    few = _analysis("Wenige Beispiele")
    assert few.metrics["decided"] <= 0.1 and few.metrics["decided"] < a.metrics["decided"] - 0.2

    flex = _analysis("Flexible Randfunktionen")
    assert flex.metrics["decided"] <= a.metrics["decided"] - 0.2

    nl = _analysis("Nicht additiv")
    assert nl.metrics["wrong_n"] >= 1 and _analysis("Nicht additiv", decider="roh_linear").metrics["wrong_n"] == 0

    noisy = _analysis("Verrauschte Urteile")
    assert not noisy.case.compatible and _analysis("Verrauschte Urteile", sigma=0.0).case.compatible
