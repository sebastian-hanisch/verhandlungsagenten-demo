"""Randfälle: kleinste Domänen (n = 2, m = 2), Frist T = 2, Reservationsnutzen 0 und so hoch, dass nichts rational ist, völliger Gleichstand, inklusive Grenzen, Slider-Maximum."""

import numpy as np
import pytest

import vha_domain as D
import vha_evaluation as V
import vha_negotiation as NG
import vha_strategies as ST
from vha_scenario import Settings


def run(dom, a, b, horizon, rule=True):
    return NG.negotiate(dom, NG.make_agent(a), NG.make_agent(b), horizon, rule)


def test_smallest_domain_two_points_two_values_all_pairs_run():
    d = D.make_domain(1, n=2, m=2)
    assert d.N == 4
    for a in ST.POOL:
        for b in ST.POOL:
            r = run(d, a, b, 6)
            assert 1 <= r.t <= 6 and r.u0 >= d.r[0] - 1e-12 and r.u1 >= d.r[1] - 1e-12


def test_horizon_two_has_one_offer_each_and_the_second_target_is_the_reservation():
    d = D.make_domain(2)
    tr = []
    r = NG.negotiate(d, NG.make_agent("Linear"), NG.make_agent("Linear"), 2, True, trace=tr)
    assert [row[0] for row in tr] == ([1, 2] if len(tr) == 2 else [1])
    assert [round(row[4], 12) for row in tr][:1] == [1.0]
    assert (not r.agree) or r.t in (1, 2)
    if len(tr) == 2:
        assert tr[1][4] == pytest.approx(float(d.r[1]))       # Ziel der zweiten Position in der letzten Runde: der Reservationsnutzen


def test_in_a_two_round_negotiation_the_last_round_agrees_exactly_when_the_offer_is_rational_for_the_responder():
    """T = 2, Regel an: in Runde 2 nimmt Position 0 jedes Angebot ab r an (und nichts darunter); Einigung in Runde 1 hat Vorrang."""
    for seed in range(30):
        d = D.make_domain(D.seed_of(f"t2|{seed}"))
        tr = []
        res = NG.negotiate(d, NG.make_agent("Hartnäckig"), NG.make_agent("Linear"), 2, True, trace=tr)
        if tr[0][3]:
            assert res.agree and res.t == 1
        else:
            assert len(tr) == 2 and res.agree == bool(d.u[0][tr[1][2]] >= d.r[0] - 1e-12)


def test_reservation_zero_makes_every_bundle_rational():
    d = D.make_domain(4, r=0.0)
    info = D.dom_info(d)
    assert info["nir"] == d.N and info["nash"] == int(np.argmax(d.u[0] * d.u[1]))


def test_reservation_one_leaves_no_rational_bundle():
    d = D.make_domain(D.seed_of("app|35"), n=4, m=4, s=1.0, g=1.0, r=1.0)
    assert D.dom_info(d) is None


def test_no_rational_bundle_is_reported_without_a_crash(monkeypatch):
    """Erzwingt den Fall „kein rationales Bündel“ in der App-Auswertung."""
    monkeypatch.setattr(D, "dom_info", lambda dom: None)
    view = V.analyse(Settings())
    kind, text = V.verdict_negotiation(view)
    assert kind == "warning" and "kein Bündel" in text and V.summary_rows(view) == []


def test_complete_tie_all_bundles_equal_for_both_sides():
    """Alle Wertlisten gleich 1: jedes Bündel hat Nutzen 1 für beide. Die erste Antwort ist sofort angenommen."""
    n, m = 3, 3
    w = [[1 / 3] * 3, [1 / 3] * 3]
    v = [np.ones((n, m)), np.ones((n, m))]
    d = D.Domain(w, v, [0.2, 0.2], np.arange(m ** n) / 100.0)
    r = run(d, "Boulware", "Conceder", 10)
    assert r.agree and r.t == 1 and r.offer == 0              # kleinster tie-Wert: Bündel 0
    info = D.dom_info(d)
    assert len(info["front"]) == d.N and info["nash"] == 0      # alle Paare gleich: alle auf der Grenze; Nash: erster Index


def test_window_edges_are_inclusive():
    u = np.array([0.5, 0.6, 0.65, 0.7])
    tie = np.arange(4) / 10
    assert ST.window_choice(u, 0.6, 0.05, None, tie) == 1          # u = 0,6 genau am unteren Rand
    assert ST.window_choice(u, 0.6, 0.05, np.array([0, 0, 1.0, 0]), tie) == 2     # u = 0,65 genau am oberen Rand (0,6 + 0,05)


def test_slider_maximum_five_points_five_values_hundred_rounds_runs():
    d = D.make_domain(5, n=5, m=5)
    assert d.N == 3125
    r = run(d, "Boulware + Modell + Endspiel", "Boulware + Modell", 100)
    assert 1 <= r.t <= 100
    view = V.analyse(Settings(n=5, m=5, T=100, first="Boulware + Modell", second="Tit-for-Tat (relativ)"))
    assert V.trace_rows(view) and V.model_trace(view)


def test_every_pair_of_the_seven_strategies_runs_at_the_extremes():
    for kw in (dict(n=3, m=3, T=2, s=0.0, g=0.0, r=0.0), dict(n=5, m=5, T=30, s=1.0, g=1.0, r=0.5), dict(n=3, m=5, T=3, s=0.5, g=0.5, r=0.2)):
        d = D.make_domain(7, n=kw["n"], m=kw["m"], s=kw["s"], g=kw["g"], r=kw["r"])
        for a in ST.POOL:
            for b in ST.POOL:
                res = NG.negotiate(d, NG.make_agent(a), NG.make_agent(b), kw["T"])
                assert res.t <= kw["T"] and (res.offer >= 0) == res.agree


def test_trace_rows_and_summary_for_the_default_settings():
    view = V.analyse(Settings())
    rows = V.trace_rows(view)
    assert rows[0]["Runde"] == "1" and rows[0]["Bietet"] == "Verlader" and rows[1]["Bietet"] == "Spediteur"
    assert rows[-1]["Antwort"] == ("angenommen" if view.result.agree else "abgelehnt")
    assert any(r["Kennzahl"] == "Ausgang" for r in V.summary_rows(view))
