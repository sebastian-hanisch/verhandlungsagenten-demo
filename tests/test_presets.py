"""Voreinstellungen: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset einzeln erkennbar; je Preset die erwartete Verdikt-Zeile (§11a Punkt 7)."""


import vha_constants as C
import vha_evaluation as V
import vha_presets as P
import vha_strategies as ST
from vha_scenario import Settings, build_domain


def _settings(preset: dict) -> Settings:
    return Settings(n=preset["n"], m=preset["m"], s=preset["s"], g=preset["g"], r=preset["r"], T=preset["T"], band=preset["band"], rule=preset["rule"], first=preset["first"],
                    second=preset["second"], seed=preset["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert _settings(next(iter(C.PRESETS.values()))) == Settings()


def test_setting_specs_cover_every_preset_key_and_defaults_match_base():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == C.BASE[key]


def test_presets_are_pairwise_different_so_that_each_can_be_highlighted():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_strategy_choices_are_the_seven_named_strategies():
    assert P.STRATEGY_CHOICES == tuple(ST.POOL) and len(P.STRATEGY_CHOICES) == 7


def _view(name):
    return V.analyse(_settings(C.PRESETS[name]))


def test_every_preset_gives_the_expected_verdict_line():
    """Erwartete Ausgänge der acht Voreinstellungen (gemessen im Bau, feste Seeds): Anfang des Verdikts je Voreinstellung."""
    expected = {
        "Boulware gegen Tit-for-Tat (Voreinstellung)": "Einigung in Runde 19 von 20: Verlader (Boulware) 0,424, Spediteur (Tit-for-Tat (relativ)) 0,850",
        "Zwei Sturköpfe": "Einigung in Runde 20 von 20, in der letzten Runde: Verlader (Hartnäckig) 0,254, Spediteur (Hartnäckig) 1,000",
        "Kurze Frist (T = 6)": "Einigung in Runde 6 von 6, in der letzten Runde: Verlader (Boulware) 0,371, Spediteur (Tit-for-Tat (relativ)) 0,934",
        "Lange Frist (T = 60)": "Einigung in Runde 54 von 60: Verlader (Boulware) 0,637, Spediteur (Boulware + Modell) 0,536",
        "Gegner bietet zuletzt": "Einigung in Runde 20 von 20, in der letzten Runde: Verlader (Hartnäckig) 0,493, Spediteur (Linear) 0,214",
        "Modell wird getäuscht": "Einigung in Runde 16 von 20: Verlader (Boulware + Modell) 0,823, Spediteur (Conceder) 0,246",
        "Gleiche Gewichte": "Einigung in Runde 19 von 20: Verlader (Boulware) 0,443, Spediteur (Tit-for-Tat (relativ)) 0,977",
        "Endspiel-Ausnutzer": "Einigung in Runde 20 von 20, in der letzten Runde: Verlader (Boulware) 0,254, Spediteur (Boulware + Modell + Endspiel) 0,892",
    }
    assert set(expected) == set(C.PRESETS)
    for name, start in expected.items():
        kind, text = V.verdict_negotiation(_view(name))
        assert kind == "info" and text.startswith(start), (name, text)


def test_two_stubborn_sides_agree_only_through_the_last_round_rule():
    """Der Hilfetext sagt: „mit ihr aus: keine Einigung“. Das Verdikt nennt die Folge der Regel, und mit der Regel aus gibt es tatsächlich keine Einigung."""
    kind, text = V.verdict_negotiation(_view("Zwei Sturköpfe"))
    assert "Mit der Regel „letzte Runde“ aus wäre es anders gekommen: keine Einigung" in text
    p = {**C.PRESETS["Zwei Sturköpfe"], "rule": "aus"}
    kind, text = V.verdict_negotiation(V.analyse(_settings(p)))
    assert kind == "warning" and text.startswith("Keine Einigung in 20 Runden") and "Mit der Regel „letzte Runde“ an wäre es anders gekommen: Einigung in Runde 20." in text


def test_the_stubborn_first_against_linear_preset_needs_the_rule():
    """Hilfetext: ohne die Regel „letzte Runde“ gäbe es in dieser Domäne keine Einigung."""
    _, text = V.verdict_negotiation(_view("Gegner bietet zuletzt"))
    assert "Mit der Regel „letzte Runde“ aus wäre es anders gekommen: keine Einigung" in text
    view = _view("Gegner bietet zuletzt")
    assert view.result.agree and view.result.t == 20 and view.trace[-1][1] == 1       # der Spediteur bietet zuletzt


def test_short_horizon_preset_help_matches_the_tournament():
    """Hilfetext „In der Turnierwertung (E2) führt bei T = 6 Tit-for-Tat“: wahr für alle drei Gewichtsunterschiede der Messreihen."""
    import vha_experiments as XP
    import vha_results as R
    data = XP.load()
    assert all(R.winners(data)[(6, s)][0] == ST.TFT for s in C.TURNIER_SS)


def test_the_deceived_model_preset_shows_a_negative_rank_correlation_for_the_loader():
    rows = [r for r in V.model_trace(_view("Modell wird getäuscht")) if r["Beobachter"] == "Verlader"]
    assert rows and rows[-1]["rho"] < 0 and rows[-1]["k"] == 8


def test_the_endgame_preset_gives_the_forwarder_more_than_boulware_plus_model_in_that_domain():
    st = _settings(C.PRESETS["Endspiel-Ausnutzer"])
    with_end = V.analyse(st).result
    without = V.analyse(Settings(**{**st.__dict__, "second": ST.BOULWARE_MODELL})).result
    assert with_end.agree and with_end.t == st.T and with_end.u1 > without.u1 + 0.05


def test_equal_weights_preset_has_zero_weight_gap():
    import vha_domain as D
    assert D.weight_gap(build_domain(_settings(C.PRESETS["Gleiche Gewichte"]))) < 1e-12


def test_the_preset_domains_have_a_rational_bundle():
    import vha_domain as D
    for name, preset in C.PRESETS.items():
        assert D.dom_info(build_domain(_settings(preset))) is not None, name
