"""Zweig-Tests (DEMO-PLAYBOOK §0, §1b): jede abschaltbare Einheit weicht auf mindestens einer Instanz vom Nachbarn ab (sonst läuft die Variante nie), dokumentierte Gleichheiten und tote Zweige."""

import pytest

import vha_domain as D
import vha_negotiation as NG
import vha_strategies as ST

S = ST.Spec
DOMS = [D.make_domain(D.seed_of(f"branch|{k}")) for k in range(40)]


def _count(spec_a, spec_b, horizon=20, rule=True, band=NG.BAND, opponents=ST.POOL6):
    """Zahl der (Domäne, Gegner, Position)-Fälle, in denen die beiden Spezifikationen verschiedene Ergebnisse liefern, und die Zahl aller Fälle."""
    diff = total = 0
    for d in DOMS:
        for o in opponents:
            for pos in (0, 1):
                out = []
                for spec in (spec_a, spec_b):
                    me, op = NG.make_agent(spec), NG.make_agent(o)
                    out.append(NG.negotiate(d, me, op, horizon, rule, band) if pos == 0 else NG.negotiate(d, op, me, horizon, rule, band))
                total += 1
                diff += out[0] != out[1]
    return diff, total


def test_total_cases_are_480():
    assert _count(S(), S())[1] == 480


@pytest.mark.parametrize("a,b", [
    (S(curve="zeit", beta=0.2, ac="zeit"), S(curve="zeit", beta=0.2, ac="next")),          # Zeitplan gegen nächstes Gebot (Boulware)
    (S(curve="zeit", beta=1.0, ac="zeit"), S(curve="zeit", beta=1.0, ac="next")),          # dieselben Regeln bei Linear weichen nur selten ab
    (S(curve="zeit", beta=1.0, ac="schwelle"), S(curve="zeit", beta=1.0, ac="reserv")),
    (S(curve="zeit", beta=0.2, model="keins"), S(curve="zeit", beta=0.2, model="haeufig")),
    (S(curve="zeit", beta=0.2, model="haeufig"), S(curve="zeit", beta=0.2, model="orakel")),
    (S(curve="zeit", beta=0.2, model="haeufig"), S(curve="zeit", beta=0.2, model="haeufig", endgame=True)),
    (S(curve="zeit", beta=0.2, model="orakel"), S(curve="zeit", beta=0.2, model="orakel", endgame=True)),
    (S(curve="tft", gamma=1.0), S(curve="tft", gamma=0.5)),
    (S(curve="zeit", beta=0.2), S(curve="zeit", beta=5.0)),
])
def test_each_switchable_unit_changes_the_outcome_on_at_least_one_instance(a, b):
    diff, total = _count(a, b)
    assert 0 < diff <= total


def test_the_endgame_flag_without_a_model_is_a_dead_branch_by_design():
    """Das Endspiel braucht eine Schätzung des Gegners: ohne Modell (kein Modell, keine Schätzung) ändert das Kennzeichen nichts. Die App bietet Endspiel nur zusammen mit dem Häufigkeitsmodell an."""
    diff, _ = _count(S(curve="zeit", beta=0.2), S(curve="zeit", beta=0.2, endgame=True))
    assert diff == 0
    assert ST.POOL[ST.ENDSPIEL].model == "haeufig"


def test_window_width_changes_the_outcome():
    diffs = 0
    for d in DOMS:
        for o in ST.POOL6:
            a = NG.negotiate(d, NG.make_agent("Boulware"), NG.make_agent(o), 20, True, 0.02)
            b = NG.negotiate(d, NG.make_agent("Boulware"), NG.make_agent(o), 20, True, 0.10)
            diffs += a != b
    assert diffs > 0


def test_last_round_rule_changes_the_outcome_of_the_stubborn_one():
    diffs = 0
    for d in DOMS:
        a = NG.negotiate(d, NG.make_agent("Hartnäckig"), NG.make_agent("Linear"), 20, True)
        b = NG.negotiate(d, NG.make_agent("Hartnäckig"), NG.make_agent("Linear"), 20, False)
        diffs += a != b
    assert diffs > 0


def test_tit_for_tat_against_tit_for_tat_or_stubborn_equals_stubborn_against_stubborn():
    """Dokumentierte Gleichheit (E2): Tit-for-Tat gibt nur nach, wenn der andere nachgibt; im Selbstspiel und gegen Hartnäckig bleibt sein Ziel 1, also ist es Hartnäckig."""
    for d in DOMS:
        h = NG.negotiate(d, NG.make_agent("Hartnäckig"), NG.make_agent("Hartnäckig"), 20)
        assert h == NG.negotiate(d, NG.make_agent(ST.TFT), NG.make_agent(ST.TFT), 20)
        assert h == NG.negotiate(d, NG.make_agent(ST.TFT), NG.make_agent("Hartnäckig"), 20)
        assert h == NG.negotiate(d, NG.make_agent("Hartnäckig"), NG.make_agent(ST.TFT), 20)


def test_never_acceptance_is_a_valid_observer_setting():
    d = DOMS[0]
    a0 = NG.Agent(S(ac="never"))
    a1 = NG.Agent(S(ac="never"))
    r = NG.negotiate(d, a0, a1, 6, False)
    assert not r.agree and len(a1.opp) == 3 and len(a0.opp) == 3
