"""Aushandlung gegen Handläufe an der Mini-Domäne, Regel „letzte Runde“, Fristparität, Häufigkeitsmodell von Hand."""

import numpy as np
import pytest

import vha_domain as D
import vha_negotiation as NG
import vha_opponent as OPP
import vha_strategies as ST


def run(dom, a, b, horizon, rule=True, trace=None):
    return NG.negotiate(dom, NG.make_agent(a), NG.make_agent(b), horizon, rule, trace=trace)


def test_hand_run_linear_against_linear_agrees_in_round_2(mini):
    """T = 3, Linear gegen Linear. Runde 1: Position 0 bietet Ziel 1 → b0 (u₀ = 1, u₁ = 0); Position 1 lehnt ab (u₁ = 0 liegt unter r = 0,2).
    Runde 2: Position 1 bietet Ziel 0,6 (τ = 0,5), Fenster [0,6; 0,65] in u₁: nur b7 (0,65; u₀(b7) = 0,2). Position 0 nimmt an: 0,2 ≥ r, und ihr nächstes eigenes Gebot
    (Runde 3, Ziel r = 0,2, Fenster [0,2; 0,25]) ist b7 selbst (Nutzen 0,2). Ergebnis (0,2; 0,65) in Runde 2."""
    tr = []
    res = run(mini, "Linear", "Linear", 3, trace=tr)
    assert res.agree and res.t == 2 and res.offer == 7
    assert (res.u0, res.u1) == pytest.approx((0.2, 0.65), abs=1e-12)
    assert [(t, p, o, acc) for t, p, o, acc, _ in tr] == [(1, 0, 0, False), (2, 1, 7, True)]
    assert [round(g, 6) for *_, g in tr] == [1.0, 0.6]       # Ziele der Bietenden: Runde 1 (τ = 0) 1, Runde 2 (τ = 1/2) 0,6


def test_hand_run_stubborn_first_against_linear_fails(mini):
    """Hartnäckig bietet immer b0 (u₁ = 0 < r: Linear lehnt ab, auch in der letzten Runde, weil unter dem Reservationsnutzen nie angenommen wird); Linear bietet b8/b7/…:
    Hartnäckig nimmt nur an, wenn das Angebot so viel bringt wie b0 (1,0): nie. Keine Einigung, beide bekommen r = 0,2."""
    res = run(mini, "Hartnäckig", "Linear", 3)
    assert not res.agree and (res.u0, res.u1) == (0.2, 0.2) and res.offer == -1


def test_hand_run_linear_first_against_stubborn_agrees_in_the_last_round_by_the_rule(mini):
    """Linear bietet in Runde 1 b0 und in Runde 3 (Ziel 0,2) b7 (Nutzen 0,2 für Position 0, 0,65 für Position 1). Hartnäckig (Position 1) bietet b8 (u₀ = 0).
    Runde 1: u₁(b0) = 0 < r: abgelehnt. Runde 2: Position 0 prüft b8: u₀ = 0 < r: abgelehnt. Runde 3: Position 0 bietet das Ziel 0,2 → b7 (0,2); Position 1 nimmt in der letzten Runde an
    (u₁ = 0,65 ≥ r, Regel an), obwohl 0,65 weniger ist als ihr Ziel 1,0. Ergebnis (0,2; 0,65) in Runde 3; ohne die Regel keine Einigung (0,65 < 1)."""
    res = run(mini, "Linear", "Hartnäckig", 3)
    assert res.agree and res.t == 3 and (res.u0, res.u1) == pytest.approx((0.2, 0.65), abs=1e-12)
    res_off = run(mini, "Linear", "Hartnäckig", 3, rule=False)
    assert not res_off.agree and (res_off.u0, res_off.u1) == (0.2, 0.2)


def test_last_round_rule_never_accepts_below_the_reservation(mini):
    """Die Regel „letzte Runde“ nimmt alles ab r an, aber nichts darunter: Hartnäckig (b0, u₁ = 0) scheitert bei T = 2 auch an einem Gegner, der in Runde 1 gehört wird."""
    res = run(mini, "Hartnäckig", "Hartnäckig", 2)
    assert not res.agree


def test_parity_decides_who_bids_last(mini):
    """T gerade: Position 1 bietet zuletzt; T ungerade: Position 0. Der Hartnäckige gewinnt, wenn der Gegner zuletzt bietet (Regel an)."""
    tr = []
    run(mini, "Hartnäckig", "Linear", 4, trace=tr)
    assert tr[-1][1] == 1 and len(tr) == 4
    tr = []
    run(mini, "Hartnäckig", "Linear", 5, trace=tr)
    assert tr[-1][1] == 0 and len(tr) == 5


def test_the_one_who_does_not_bid_last_accepts_in_the_final_round_by_the_rule(mini):
    """T = 4, Hartnäckig als Position 0: Linear bietet zuletzt (Runde 4, Ziel r = 0,2, Fenster [0,2; 0,25] in u₁ leer, kleinster Nutzen über 0,2 ist 0,3 = b6 mit u₀ = 0,4).
    Hartnäckig nimmt in der letzten Runde an (u₀ = 0,4 ≥ r, Regel an), obwohl sein nächstes Gebot 1 wäre: Einigung in Runde 4 bei (0,4; 0,3). Ohne die Regel keine Einigung."""
    res = run(mini, "Hartnäckig", "Linear", 4)
    assert res.agree and res.t == 4 and res.offer == 6 and (res.u0, res.u1) == pytest.approx((0.4, 0.3), abs=1e-12)
    assert not run(mini, "Hartnäckig", "Linear", 4, rule=False).agree


def test_result_fields_and_trace_for_an_agreement_in_round_one():
    """Beide Seiten gleich bewertet: u₁ = u₀. Dann akzeptiert Position 1 das erste Gebot (u = 1) sofort."""
    d = D.make_domain(3, n=3, m=3, s=0.0, g=0.0)
    same = D.Domain([d.w[0], d.w[0]], [d.v[0], d.v[0]], d.r, d.tie)
    tr = []
    res = run(same, "Boulware", "Linear", 20, trace=tr)
    assert res.agree and res.t == 1 and res.u0 == pytest.approx(1.0) and res.u1 == pytest.approx(1.0) and len(tr) == 1 and tr[0][3] is True


def test_evaluate_distance_and_relative_nash_product(mini):
    info = D.dom_info(mini)
    ev = NG.evaluate(mini, info, NG.Result(True, 2, 2, 0.6, 0.7))        # Nash-Bündel b2 selbst
    assert ev["dist"] == pytest.approx(0.0, abs=1e-12) and ev["nashrel"] == pytest.approx(1.0) and ev["wel"] == pytest.approx(1.3)
    ev = NG.evaluate(mini, info, NG.Result(True, 4, 4, 0.5, 0.5))        # b4 (0,5; 0,5): Produkt 0,09/0,20 = 0,45; Abstand zu (0,6; 0,7): √(0,01 + 0,04) = 0,2236
    assert ev["nashrel"] == pytest.approx(0.45) and ev["dist"] == pytest.approx(np.sqrt(0.05), abs=1e-9)
    ev = NG.evaluate(mini, info, NG.Result(False, 3, -1, 0.2, 0.2))      # keine Einigung: Nash-Produkt 0; Abstand zur nächsten Ecke (0,6; 0,7)? nächste Ecke ist b1 (0,8; 0,35): √(0,36 + 0,0225)
    assert ev["nashrel"] == 0.0 and not ev["agree"]
    assert ev["dist"] == pytest.approx(min(np.hypot(0.2 - a, 0.2 - b) for a, b in [(0.8, 0.35), (0.6, 0.7), (0.3, 0.85)]), abs=1e-12)


def test_frequency_model_by_hand_offers_2_2_5(mini):
    """Angebote b2 = (0; 2), b2, b5 = (1; 2): Zähler Punkt 0: [2, 1, 0], Punkt 1: [0, 0, 3]; gleich geblieben: Punkt 0 einmal, Punkt 1 zweimal → Gewichte (2; 3)/5 = (0,4; 0,6).
    Wertschätzung Punkt 0: (3; 2; 1)/3, Punkt 1: (1; 1; 4)/4. û(b2) = 0,4·1 + 0,6·1 = 1; û(b5) = 0,4·(2/3) + 0,6·1 = 0,8667; û(b0) = 0,4·1 + 0,6·0,25 = 0,55."""
    ag = NG.make_agent("Boulware + Modell")
    ag.reset(mini, 0, 5)
    for o in (2, 2, 5):
        ag.observe(o)
    assert ag.counts.tolist() == [[2, 1, 0], [0, 0, 3]] and ag.nsame.tolist() == [1, 2]
    assert ag.est_weights().tolist() == pytest.approx([0.4, 0.6], abs=1e-12)
    e = ag.est()
    assert e[2] == pytest.approx(1.0) and e[5] == pytest.approx(0.4 * 2 / 3 + 0.6) and e[0] == pytest.approx(0.4 + 0.6 * 0.25)
    assert (e[2], e[5], e[0]) == pytest.approx((1.0, 0.86667, 0.55), abs=1e-5)


def test_model_choice_picks_the_bundle_the_opponent_seems_to_like(mini):
    """Ziel 0,55 mit Fenster 0,15 ergibt {b2, b3}. Ohne Modell entscheidet die Gleichstandsregel (kleinerer tie-Wert: b2). Nach den Angeboten b3, b3, b3 (Punkt 0 Wert 1, Punkt 1 Wert 0)
    schätzt das Modell û(b3) = 1 (Wertschätzung 1 an beiden Punkten) und û(b2) = 0,5·(1/4) + 0,5·... kleiner; es wählt b3."""
    ag = NG.make_agent("Boulware + Modell")
    ag.reset(mini, 0, 5, band=0.15)
    for o in (3, 3, 3):
        ag.observe(o)
    e = ag.est()
    assert e[3] == pytest.approx(1.0) and e[2] < e[3]
    assert ag.choose(0.55) == 3
    plain = NG.make_agent("Boulware")
    plain.reset(mini, 0, 5, band=0.15)
    for o in (3, 3, 3):
        plain.observe(o)
    assert plain.choose(0.55) == 2          # ohne Modell entscheidet die Gleichstandsregel (kleinerer Index)


def test_oracle_model_uses_the_true_utility(mini):
    ag = NG.Agent(ST.Spec(model="orakel"))
    ag.reset(mini, 0, 5)
    assert ag.est() is mini.u[1] or np.array_equal(ag.est(), mini.u[1])


def test_rank_correlation_known_values():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert OPP.rank_correlation(a, a) == pytest.approx(1.0) and OPP.rank_correlation(a, -a) == pytest.approx(-1.0)
    assert OPP.rank_correlation(a, np.ones(4)) == 0.0                                    # konstante Seite: nicht definiert, hier 0
    # Gleichstände: Ränge (1; 2,5; 2,5; 4) gegen (1; 2; 3; 4): r = 0,9487
    assert OPP.rank_correlation(np.array([1.0, 2.0, 2.0, 3.0]), a) == pytest.approx(0.9486833, abs=1e-6)


def test_rank_correlation_matches_scipy_when_available():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = rng.integers(0, 6, 30).astype(float), rng.random(30)
        assert OPP.rank_correlation(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-12)


def test_weight_error_is_half_l1():
    assert OPP.weight_error(np.array([0.5, 0.5]), np.array([0.25, 0.75])) == pytest.approx(0.25)
