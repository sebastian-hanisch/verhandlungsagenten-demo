"""Verdikte als reine Funktionen: Art und jede Zahl im Text an handgesetzten Werten bzw. gegen die Rechnung (DEMO-PLAYBOOK §12, Verdikte als reine Funktionen)."""

import numpy as np
import pytest

import vha_constants as C
import vha_evaluation as V
import vha_experiments as XP
import vha_results as R
import vha_strategies as ST
from vha_scenario import Settings


@pytest.fixture(scope="module")
def data():
    d = XP.load()
    assert d is not None
    return d


def test_verdict_domain_numbers_and_the_largest_gap_by_hand():
    rows = [{"s": 0.0, "nash_sum": 1.2, "max_sum": 1.25}, {"s": 0.5, "nash_sum": 1.3, "max_sum": 1.32}, {"s": 1.0, "nash_sum": 1.5, "max_sum": 1.52}]
    kind, text = V.verdict_domain(rows, 400)
    assert kind == "info"
    for needle in ("400 Zufallsdomänen", "von 1,200 bei gleichen Gewichten (s = 0,0) auf 1,500", "von 1,250 auf 1,520", "höchstens 0,050 unter der größten Summe"):
        assert needle in text, needle


def test_verdict_negotiation_names_both_strategies_the_round_and_the_rule_effect():
    kind, text = V.verdict_negotiation(V.analyse(Settings()))
    assert kind == "info" and "Verlader (Boulware)" in text and "Spediteur (Tit-for-Tat (relativ))" in text and "Runde 19 von 20" in text
    assert "Mit der Regel" not in text                      # die Regel ändert an diesem Ausgang nichts


def test_verdict_negotiation_no_agreement_branch_gives_the_reservation_and_the_largest_sum():
    st = Settings(first="Hartnäckig", second="Hartnäckig", rule="aus")
    view = V.analyse(st)
    kind, text = V.verdict_negotiation(view)
    assert kind == "warning" and f"Reservationsnutzen {V.de(st.r, 2)}" in text and f"{V.de(view.info['maxsum'], 2)}" in text and "Summe 0,40" in text


def test_verdict_model_live_empty_and_filled():
    view = V.analyse(Settings())
    assert V.verdict_model_live(view, [])[1] == "Es gab keine Angebote des Gegners zu beobachten."
    rows = V.model_trace(view)
    kind, text = V.verdict_model_live(view, rows)
    last_v = [r for r in rows if r["Beobachter"] == "Verlader"][-1]
    assert kind == "info" and f"nach {last_v['k']} Angeboten des Gegners: Gewichtsabstand {V.de(last_v['gap'], 3)}" in text and f"{len(C.SEEDS) * C.MODEL_DOMAINS} Domänen" in text


def test_verdict_turnier_numbers_are_the_computed_ones(data):
    _, text = V.verdict_turnier(data)
    c = R.turnier_cell(data, 20, 0.5)
    f = V.top_group_facts(data)
    last_mean, last_seed, exc = R.last_place_counts(data)
    assert f"Conceder liegt in {last_mean} von 9 Zellen" in text and f"in {last_seed} von 36 Zelle-Seed-Paaren" in text
    assert f"Boulware + Modell {V.de(c.mean['own'][5], 3)}, Boulware {V.de(c.mean['own'][1], 3)}, Tit-for-Tat {V.de(c.mean['own'][4], 3)}" in text
    assert f"in {f['ahead']} von 9 Zellen vor Boulware" in text and f"zwischen {V.de(f['min'], 3)} und {V.de(f['max'], 3)}" in text


def test_verdict_position_numbers(data):
    _, text = V.verdict_position(data)
    f = V.position_facts(data)
    for key, fmt in (("t20_first", "mine"), ("t20_second", "mine"), ("t21_first", "mine"), ("t21_second", "mine"), ("t20_first_off", "mine"), ("t20_second_off", "mine")):
        assert V.de(f[key][fmt], 3) in text, key
    assert f["t20_first"]["last"] is False and f["t20_second"]["last"] is True and f["t21_first"]["last"] is True and f["t21_second"]["last"] is False


def test_verdict_rule_says_whether_the_order_of_the_other_five_stays(data):
    _, text = V.verdict_rule(data)
    on, off = R.turnier_cell(data, 20, 0.5), R.rule_off_cell(data)
    rest = [i for i in range(6) if i != 0]
    same = list(np.argsort(-on.mean["own"][rest])) == list(np.argsort(-off.mean["own"][rest]))
    assert ("ihre Rangfolge bleibt" in text) == same
    assert f"von {V.de(on.mean['own'][0], 3)} auf {V.de(off.mean['own'][0], 3)}" in text


def test_verdict_endgame_numbers(data):
    _, text = V.verdict_endgame(data)
    c7 = R.endgame_cell(data, 20, "7")
    assert V.de(c7.mean["own"][6], 3) in text and f"{V.de(c7.mean['agree'][5], 3)} auf {V.de(c7.mean['agree'][6], 3)}" in text
    assert f"Rang 1 in {int((c7.ranks[:, 6] == 1).sum())} von 4 Seeds" in text


def test_verdict_model_numbers(data):
    _, text = V.verdict_model(data)
    lin, con = R.model_curve(data, ST.LINEAR, 0.5), R.model_curve(data, ST.CONCEDER, 0.5)
    assert V.de(lin["gap"][-1], 3) in text and V.de(lin["guess"], 3) in text and V.signed(con["rho"][-1], 3) in text
    assert V.pct(R.effect_gap_closed(data, "Boulware", "wel"), 0) in text


def test_verdict_decomposition_names_the_stage_dependence(data):
    _, text = V.verdict_decomp(data)
    e = V.eta_values(data, "alle 48 Agenten", "own")
    assert f"Akzeptanz {V.de(e['Akzeptanz'][0], 3)}, Kurve {V.de(e['Kurve'][0], 3)}, Modell {V.de(e['Modell'][0], 3)}" in text
    assert "Eine feste Rangfolge Gebot vor Annahme vor Modell ergibt sich in diesem Modell nicht." in text


def test_verdict_robust_counts(data):
    _, text = V.verdict_robust(data)
    rc = V.robust_counts(data)
    assert f"gewinnt " in text and f"Conceder ist in {rc['conceder_last']} von {rc['n']} Fällen Letzter" in text
    assert sum(len(v) for v in rc["winners"].values()) == rc["n"] == 18


def test_verdict_learner_is_a_warning_with_the_bands(data):
    kind, text = V.verdict_learner(data)
    s = [R.learner_summary(data, E) for E in C.LEARN_EPISODES]
    assert kind == "warning" and f"{V.de(s[0]['mean'], 3)} / {V.de(s[1]['mean'], 3)} / {V.de(s[2]['mean'], 3)}" in text and "über andere Gegner sagt die Messung nichts" in text


def test_table_builders_have_the_documented_columns(data):
    assert list(V.turnier_table(R.turnier_cell(data, 20, 0.5))[0]) == ["Strategie", "Eigener Nutzen (Std)", "Summe", "Abstand zur Grenze", "Einigungsrate", "Nash-Produkt rel.", "Rang je Seed"]
    assert len(V.robust_rows(data)) == 18 and len(V.position_table(data)) == 4 * 2 * 3 and len(V.rule_rows(data)) == 6
    assert len(V.learner_rows(data)) == 7 and len(V.transfer_rows(data)) == 6 and len(V.omitted_rows(data)) == 2


def test_formats():
    assert V.de(0.5) == "0,50" and V.de(-0.25, 3) == "−0,250" and V.de(-0.0001, 2) == "0,00"
    assert V.pct(0.123, 1) == "12,3 %" and V.signed(0.5, 3) == "+0,500" and V.signed(-0.5, 3) == "−0,500" and V.signed(0.0, 3) == "0,000"
    assert V.thousands(3125) == "3.125" and V.pm(0.5, 0.01) == "0,500 (0,010)"


def test_first_bid_table_by_hand_for_the_stubborn_and_the_linear_strategy():
    """Hartnäckig bietet immer das Bündel mit Nutzen 1; Linear bietet in Runde 3 (T = 20, r = 0,2) ein Bündel im Fenster [Ziel, Ziel + 0,05] mit Ziel 1 − 0,8 · 2/19 = 0,9158.
    Das letzte eigene Gebot des Verladers ist bei T = 20 das der Runde 19, bei T = 21 das der Runde 21."""
    rows = {r["Strategie"]: r for r in V.first_bid_rows(Settings())}
    assert len(rows) == 7 and rows["Hartnäckig"]["Runde 1"] == rows["Hartnäckig"]["Runde 7"] == "1,000" and rows["Hartnäckig"]["letztes eigenes Gebot"] == "1,000 (Runde 19)"
    target = 1 - 0.8 * 2 / 19
    lin = float(rows["Linear"]["Runde 3"].replace(",", ".").replace("−", "-"))
    assert target - 1e-3 <= lin <= target + 0.05 + 1e-3 and rows["Linear"]["Runde 1"] == "1,000"
    short_rows = {r["Strategie"]: r for r in V.first_bid_rows(Settings(T=2))}
    assert short_rows["Boulware"]["Runde 3"] == "–" and short_rows["Boulware"]["letztes eigenes Gebot"].endswith("(Runde 1)")
    assert V.first_bid_rows(Settings(T=21))[0]["letztes eigenes Gebot"].endswith("(Runde 21)")
