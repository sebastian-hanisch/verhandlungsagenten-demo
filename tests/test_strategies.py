"""Bausteine der Strategien gegen Handrechnung an der Mini-Domäne (conftest.mini): Zielkurven, Tit-for-Tat, Wahl auf der Isolinie, Annahmeregeln, Endspiel."""

import numpy as np
import pytest

import vha_negotiation as NG
import vha_strategies as ST
from conftest import MINI_U0


def test_progress_runs_from_zero_to_one():
    assert [ST.progress(t, 5) for t in (1, 2, 3, 4, 5)] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert ST.progress(0, 5) == 0.0 and ST.progress(9, 5) == 1.0       # außerhalb geklemmt
    assert ST.progress(1, 2) == 0.0 and ST.progress(2, 2) == 1.0


def test_target_curves_at_tau_one_half_by_hand():
    """T = 3, t = 2 (τ = 0,5), r = 0,2: Boulware β = 0,2: 1 − 0,8·0,5⁵ = 0,975; Linear: 1 − 0,8·0,5 = 0,6; Conceder β = 5: 1 − 0,8·0,5^0,2 = 0,30356."""
    assert ST.target_curve(2, 3, 0.2, 0.2) == pytest.approx(0.975, abs=1e-12)
    assert ST.target_curve(2, 3, 0.2, 1.0) == pytest.approx(0.6, abs=1e-12)
    assert ST.target_curve(2, 3, 0.2, 5.0) == pytest.approx(1 - 0.8 * 0.5 ** 0.2, abs=1e-12)
    assert ST.target_curve(2, 3, 0.2, 5.0) == pytest.approx(0.30356, abs=1e-5)


@pytest.mark.parametrize("beta", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
def test_target_curve_starts_at_one_ends_at_r_and_never_rises(beta):
    vals = [ST.target_curve(t, 20, 0.3, beta) for t in range(1, 21)]
    assert vals[0] == 1.0 and vals[-1] == pytest.approx(0.3, abs=1e-12)
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


def test_smaller_beta_holds_the_target_higher_at_every_round_before_the_end():
    for t in range(2, 20):
        v = [ST.target_curve(t, 20, 0.2, b) for b in (0.2, 1.0, 5.0)]
        assert v[0] > v[1] > v[2]


def test_tit_for_tat_target_falls_by_the_gain_of_the_opponent_never_below_r():
    assert ST.tft_target(1.0, 0.25, 1.0, 0.2) == pytest.approx(0.75)
    assert ST.tft_target(0.3, 0.25, 1.0, 0.2) == 0.2
    assert ST.tft_target(0.9, 0.0, 1.0, 0.2) == 0.9 and ST.tft_target(0.9, -0.3, 1.0, 0.2) == 0.9
    assert ST.tft_target(1.0, 0.4, 0.5, 0.2) == pytest.approx(0.8)


def test_window_choice_picks_inside_the_window_and_breaks_ties_by_tie_vector(mini):
    """Ziel 0,6 und Fenster 0,05: Nutzen in [0,6; 0,65] hat nur b2 (u₀ = 0,6). Ziel 0,45: [0,45; 0,5] enthält b4 (0,5) allein."""
    u, tie = mini.u[0], mini.tie
    assert ST.window_choice(u, 0.6, 0.05, None, tie) == 2
    assert ST.window_choice(u, 0.45, 0.05, None, tie) == 4
    # Fenster [0,55; 0,70]: b2 (0,6), b3 (0,7) – ohne Modell gewinnt bei Gleichstand der kleinere tie-Wert (b2)
    assert ST.window_choice(u, 0.55, 0.15, None, tie) == 2
    # mit Modell: der größere geschätzte Gegnernutzen gewinnt (b3), egal wo er im Fenster liegt
    est = np.zeros(9)
    est[3] = 1.0
    assert ST.window_choice(u, 0.55, 0.15, est, tie) == 3


def test_window_choice_fallbacks_empty_window_and_target_above_everything(mini):
    """Leeres Fenster [0,9; 0,95]: kein Nutzen dort (b0 hat 1,0): das Bündel mit dem kleinsten Nutzen über dem Ziel = b0. Ziel 1,2 über allen: das beste Bündel (b0)."""
    u, tie = mini.u[0], mini.tie
    assert ST.window_choice(u, 0.9, 0.05, None, tie) == 0
    assert ST.window_choice(u, 1.2, 0.05, None, tie) == 0
    # Fenster leer zwischen 0,35 und 0,39: kleinster Nutzen über 0,35 ist 0,4 (b6)
    assert ST.window_choice(u, 0.35, 0.04, None, tie) == 6


def test_accept_rules_by_hand():
    # Reservation 0,2; Frist 20; Angebot mit Nutzen 0,5 in Runde 10 (τ = 9/19)
    assert not ST.accept_rule("reserv", 0.19, 10, 20, 0.2, 0.7, 0.0)
    assert ST.accept_rule("reserv", 0.2, 10, 20, 0.2, 0.7, 0.0)                    # inklusive Grenze
    assert not ST.accept_rule("schwelle", 0.69, 10, 20, 0.2, 0.7, 0.0) and ST.accept_rule("schwelle", 0.7, 10, 20, 0.2, 0.7, 0.0)
    plan = 1 - 0.8 * 9 / 19                                                         # Zeitplan in Runde 10
    assert plan == pytest.approx(0.6210526, abs=1e-6)
    assert not ST.accept_rule("zeit", plan - 1e-6, 10, 20, 0.2, 0.7, 0.0) and ST.accept_rule("zeit", plan, 10, 20, 0.2, 0.7, 0.0)
    assert ST.accept_rule("next", 0.55, 10, 20, 0.2, 0.7, 0.55) and not ST.accept_rule("next", 0.54, 10, 20, 0.2, 0.7, 0.55)
    with pytest.raises(ValueError):
        ST.accept_rule("unbekannt", 0.9, 1, 20, 0.2, 0.7, 0.0)


def test_endgame_choice_is_the_best_bundle_among_those_the_opponent_still_accepts(mini):
    """Schätzung des Gegners = wahrer Nutzen u₁; Reservation des Gegners 0,2: annehmbar sind b1, b2, b4, b5, b6, b7, b8 (u₁ ≥ 0,2); davon ist für Position 0 das beste b1 (0,8)."""
    assert ST.endgame_choice(mini.u[0], mini.u[1], 0.2, mini.tie) == 1
    assert ST.endgame_choice(mini.u[0], mini.u[1], 0.9, mini.tie) == 8      # nur b8 (u₁ = 1) bleibt: Position 0 bekommt dann 0
    assert ST.endgame_choice(mini.u[0], mini.u[1], 1.5, mini.tie) is None


def test_pool_definitions():
    assert list(ST.POOL6) == ["Hartnäckig", "Boulware", "Linear", "Conceder", "Tit-for-Tat (relativ)", "Boulware + Modell"]
    assert ST.POOL7[-1] == ST.ENDSPIEL and len(ST.POOL9) == 9 and set(ST.POOL9) <= set(ST.POOL) | set(ST.EXTRA)
    assert ST.POOL["Boulware"].beta == 0.2 and ST.POOL["Conceder"].beta == 5.0 and ST.POOL["Linear"].beta == 1.0
    assert ST.spec_of("Boulware + Orakel").model == "orakel" and ST.spec_of("Boulware + Orakel + Endspiel").endgame


def test_agent_target_follows_the_curve_and_uses_the_tft_state(mini):
    ag = NG.make_agent("Tit-for-Tat (relativ)")
    ag.reset(mini, 0, 5)
    assert ag.target(1) == 1.0
    ag.observe(3)            # erstes Angebot des Gegners: kein Vergleich
    assert ag.target(2) == 1.0
    ag.observe(1)            # u₀(b1) − u₀(b3) = 0,8 − 0,7 = 0,1: Ziel 0,9
    assert ag.target(3) == pytest.approx(0.9, abs=1e-12)
    ag.observe(8)            # Rückgang (0 − 0,8): Ziel bleibt
    assert ag.target(4) == pytest.approx(0.9, abs=1e-12)
    ag.observe(0)            # Zuwachs 1,0: Ziel fällt auf r = 0,2
    assert ag.target(5) == pytest.approx(0.2, abs=1e-12)
    h = NG.make_agent("Hartnäckig")
    h.reset(mini, 0, 5)
    assert [h.target(t) for t in range(1, 6)] == [1.0] * 5 and h.bid(3) == 0


def test_u_helper_values_match_conftest(mini):
    assert mini.u[0].tolist() == pytest.approx(MINI_U0)
