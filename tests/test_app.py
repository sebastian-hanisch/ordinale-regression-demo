"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
AUTOPLAY_MARKER = "▶"  # Experiment-Buttons (rechnen rund 20 s) im Rauchtest nur gezielt auslösen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    import ord_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="n_slider").value == preset["n"] and at.slider(key="m_slider").value == preset["m"] and at.slider(key="r_slider").value == preset["r"]
        for key, field in (("gamma_select", "gamma"), ("sigma_select", "sigma"), ("decider_select", "decider")):
            assert at.selectbox(key=key).value == preset[field], (name, key)
        assert at.number_input(key="seed_input").value == preset["seed"]


def test_every_choice_of_every_selectbox_renders_without_exception():
    import ord_constants as C

    for key, choices in (("gamma_select", C.GAMMA_CHOICES), ("sigma_select", C.SIGMA_CHOICES), ("decider_select", C.DECIDERS)):
        for choice in choices:
            at = _fresh_app()
            at.selectbox(key=key).select(choice).run(timeout=TIMEOUT)
            assert not at.exception, (key, choice, [str(e) for e in at.exception])


def test_incompatible_order_shows_the_warning_instead_of_the_relations():
    at = _fresh_app()
    at.selectbox(key="sigma_select").select(0.2).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("verträglich" in w.value for w in at.warning)


def test_extreme_settings_render_without_exception():
    at = _fresh_app()
    at.slider(key="m_slider").set_value(8).run(timeout=TIMEOUT)
    at.slider(key="n_slider").set_value(20).run(timeout=TIMEOUT)
    at.slider(key="r_slider").set_value(12).run(timeout=TIMEOUT)
    at.selectbox(key="gamma_select").select(3).run(timeout=TIMEOUT)
    at.selectbox(key="decider_select").select("untergrenze").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_smallest_matrix_and_most_references_render_without_exception():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(8).run(timeout=TIMEOUT)
    at.slider(key="r_slider").set_value(12).run(timeout=TIMEOUT)       # wirksam sind nur n - 2 = 6
    assert not at.exception, [str(e) for e in at.exception]


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_experiments_run_only_after_their_own_click():
    at = _fresh_app()
    for key in ("exp_examples", "exp_model", "exp_arbitrary", "exp_sampling"):
        assert key not in at.session_state
    at.button(key="exp_sampling_start").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "exp_sampling" in at.session_state and "exp_examples" not in at.session_state
    at.button(key="exp_examples_start").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "exp_examples" in at.session_state


def test_the_other_two_experiments_run_without_exception():
    at = _fresh_app()
    at.button(key="exp_model_start").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    at.button(key="exp_arbitrary_start").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "exp_model" in at.session_state and "exp_arbitrary" in at.session_state


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "9999"
    at.query_params["m"] = "1"
    at.query_params["r"] = "99"
    at.query_params["g"] = "2"
    at.query_params["sg"] = "0.2"
    at.query_params["dc"] = "orakel"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == 20 and at.slider(key="m_slider").value == 3 and at.slider(key="r_slider").value == 12
    assert at.selectbox(key="gamma_select").value == 1 and at.selectbox(key="sigma_select").value == 0.2 and at.selectbox(key="decider_select").value == "roh_linear"
