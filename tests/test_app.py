"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Knopf (auch die ▶️-Knöpfe: Pool-Turnier und Lernender), fährt jeden Regler an seine
Grenzen, setzt alle Minima und alle Maxima zugleich, prüft Permalink und PDF. Kein Python-Fehler, insbesondere keine `StreamlitDuplicateElementId` (mehrere `st.plotly_chart` ohne eindeutiges key=)
und keine `StreamlitAPIException` bei Reglern mit berechneten Grenzen."""

import os
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import vha_constants as C
import vha_presets as P

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    at = _fresh_app()
    assert at.title and at.metric and len(at.tabs) == 6


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    assert any("Pool-Turnier" in x for x in labels) and any("Lernenden" in x for x in labels) and any("PDF" in x for x in labels)
    for index, label in enumerate(labels):
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


@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_sliders_at_minimum_or_maximum_together(edge):
    at = _fresh_app()
    for key in ("n_slider", "m_slider", "s_slider", "g_slider", "r_slider", "t_slider", "band_slider"):
        spec = P.SETTING_SPECS[key]
        at.sidebar.slider(key=key).set_value(spec.lo if edge == "min" else spec.hi)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for first in ("Hartnäckig", "Boulware + Modell + Endspiel"):
        at.sidebar.selectbox(key="first_select").set_value(first)
        at.sidebar.selectbox(key="second_select").set_value("Tit-for-Tat (relativ)")
        at.run(timeout=TIMEOUT)
        assert not at.exception, [str(e) for e in at.exception]


def test_every_pair_of_selectbox_choices_runs():
    at = _fresh_app()
    for a in P.STRATEGY_CHOICES:
        for b in (P.STRATEGY_CHOICES[0], P.STRATEGY_CHOICES[-1]):
            at.sidebar.selectbox(key="first_select").set_value(a)
            at.sidebar.selectbox(key="second_select").set_value(b)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (a, b, [str(e) for e in at.exception])
    at.sidebar.selectbox(key="rule_select").set_value("aus")
    at.run(timeout=TIMEOUT)
    assert not at.exception


def test_the_tournament_button_shows_a_table_of_six_strategies_and_the_note_about_a_sample():
    at = _fresh_app()
    next(b for b in at.button if "Pool-Turnier" in b.label).click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("eine Stichprobe mit einem Zufalls-Seed" in c.value and f"{C.LIVE_DOMAINS_DEFAULT} Domänen" in c.value for c in at.caption)
    assert any(len(df.value) == 6 and "Strategie" in df.value.columns and "Eigener Nutzen" in df.value.columns for df in at.dataframe)


def test_the_pool_of_seven_adds_the_endgame_exploiter():
    at = _fresh_app()
    at.selectbox(key="live_pool").set_value("7")
    next(b for b in at.button if "Pool-Turnier" in b.label).click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Strategie" in df.value.columns and len(df.value) == 7 and "Endspiel-Ausnutzer" in set(df.value["Strategie"]) and "Eigener Nutzen" in df.value.columns for df in at.dataframe)


def test_the_learner_button_shows_the_result_against_the_fixed_strategy():
    at = _fresh_app()
    next(b for b in at.button if "Lernenden" in b.label).click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("ein Trainings-Seed" in c.value for c in at.caption)
    assert any("Mittel über die sechs" in set(df.value["Gegner"]) for df in at.dataframe if "Gegner" in df.value)


def test_the_domain_statistics_button_shows_five_rows():
    at = _fresh_app()
    next(b for b in at.button if "Domänenstatistik" in b.label).click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any(len(df.value) == 5 and "Summe im Nash-Bündel" in df.value.columns for df in at.dataframe)


def test_changed_settings_after_a_live_run_ask_for_a_new_run():
    at = _fresh_app()
    next(b for b in at.button if "Pool-Turnier" in b.label).click().run(timeout=TIMEOUT)
    at.sidebar.slider(key="t_slider").set_value(21).run(timeout=TIMEOUT)
    assert not at.exception and any("erneut rechnen" in c.value for c in at.caption)


def test_pdf_button_then_download_button_with_a_pdf():
    at = _fresh_app()
    assert not at.get("download_button")
    next(b for b in at.button if b.key == "pdf_button").click().run(timeout=TIMEOUT)
    assert not at.exception and at.get("download_button")
    from vha_pdf_export import build_pdf
    import vha_evaluation as V
    from vha_presets import settings_from_state  # noqa: F401  (Import-Prüfung)
    from vha_scenario import Settings
    data = build_pdf(Settings(), V.analyse(Settings()), [V.verdict_negotiation(V.analyse(Settings()))])
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_permalink_parameters_set_the_controls_and_garbage_is_ignored():
    at = AppTest.from_file(APP_PATH)
    at.query_params["T"] = "6"
    at.query_params["erster"] = "Hartnäckig"
    at.query_params["zweiter"] = "unsinn"
    at.query_params["s"] = "nan"
    at.query_params["n"] = "99"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.sidebar.slider(key="t_slider").value == 6 and at.sidebar.selectbox(key="first_select").value == "Hartnäckig"
    assert at.sidebar.selectbox(key="second_select").value == C.BASE["second"] and at.sidebar.slider(key="s_slider").value == C.BASE["s"]
    assert at.sidebar.slider(key="n_slider").value == C.N_MAX


def test_missing_data_file_gives_a_warning_not_an_exception(monkeypatch, tmp_path):
    """Fehlt die Messdatei, zeigt die App eine Warnung und die Live-Teile (der Zwischenspeicher der Daten wird vorher und nachher geleert)."""
    import streamlit as st

    import vha_experiments as XP
    st.cache_resource.clear()
    monkeypatch.setattr(XP, "DATA_FILE", tmp_path / "fehlt.json")
    try:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=TIMEOUT)
        assert not at.exception, [str(e) for e in at.exception]
        assert any("data/messreihen.json" in w.value for w in at.warning)
        assert len(at.tabs) == 6 and at.metric
    finally:
        st.cache_resource.clear()


def test_seven_experiments_in_six_tabs_and_the_documented_expanders():
    at = _fresh_app()
    labels = [s.value for s in at.subheader if "Experiment E" in s.value]
    assert sorted(re.search(r"Experiment (E\d)", x).group(1) for x in labels) == [f"E{i}" for i in range(1, 8)]      # E6 steht im Reiter „Turnier“ hinter E2
    assert [t.label for t in at.tabs] == ["1 Domäne und Orakel", "2 Turnier", "3 Wer zuletzt bietet", "4 Gegnermodell", "5 Bausteine zerlegen", "6 Lernender 🧪"]
    titles = [e.label for e in at.expander]
    for needed in ("📐 Mathematische Formulierung", "Erklärtabelle: die Strategien", "Erklärtabelle: die Kennzahlen und ihre Bezugsmenge", "Erklärtabelle: was die Regler tun"):
        assert needed in titles
    assert any(s.value == "🚧 Wo die Annahmen enden" for s in at.subheader)
