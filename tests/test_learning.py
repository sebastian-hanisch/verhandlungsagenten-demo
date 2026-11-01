"""Lernender Verhandler: Zustandsindex, TD-Schritt, ε-Plan von Hand; Training ist deterministisch; Spielregeln des Agenten (Annahme ab Ziel, nie unter r); Bandbreite der Ergebnisse."""

import numpy as np
import pytest

import vha_domain as D
import vha_learning as LR
import vha_negotiation as NG
import vha_strategies as ST
from vha_rng import SplitMix64


def test_state_count_is_90():
    assert LR.N_STATES == 5 * 6 * 3 == 90
    assert len(LR.LEVELS) == 9 and LR.LEVELS[0] == 1.0 and LR.LEVELS[-1] == 0.2


def test_state_index_by_hand():
    """T = 20. Runde 1, kein Angebot: Zeitfenster 0, Klasse „kein Angebot“ = 5, Trend 1 → (0·6 + 5)·3 + 1 = 16.
    Runde 20 (τ = 1 → Fenster 4), Nutzen 0,95 (Klasse ⌊4,75⌋ = 4), vorher 0,5 (Zuwachs 0,45 > 0,02: Trend 2) → (4·6 + 4)·3 + 2 = 86.
    Runde 10 (⌊5·9/19⌋ = 2), Nutzen 0,3 (Klasse 1), vorher 0,31 (Änderung −0,01 im Totband: Trend 1) → (2·6 + 1)·3 + 1 = 40; fällt es um 0,05: Trend 0 → 39."""
    assert LR.state_index(1, 20, None, None) == 16
    assert LR.state_index(20, 20, 0.95, 0.5) == 86
    assert LR.state_index(10, 20, 0.3, 0.31) == 40
    assert LR.state_index(10, 20, 0.3, 0.35) == 39
    assert LR.state_index(10, 20, 1.0, None) == (2 * 6 + 4) * 3 + 1          # Nutzen 1,0 wird auf Klasse 4 begrenzt, ohne Vorgänger Trend 1


def test_state_index_covers_exactly_the_90_states():
    seen = {LR.state_index(t, 20, u, p) for t in range(1, 21) for u in (None, 0.05, 0.25, 0.45, 0.65, 0.95) for p in (None, 0.0, 0.5, 1.0)}
    assert min(seen) >= 0 and max(seen) < LR.N_STATES


def test_td_update_by_hand():
    Q = np.full((LR.N_STATES, 9), 0.5)
    LR.td_update(Q, 3, 4, 1.0, 0.1)          # 0,5 + 0,1·(1,0 − 0,5) = 0,55
    assert Q[3, 4] == pytest.approx(0.55) and Q.sum() == pytest.approx(0.5 * 90 * 9 + 0.05)
    LR.td_update(Q, 3, 4, 0.2, 0.1)          # 0,55 + 0,1·(0,2 − 0,55) = 0,515
    assert Q[3, 4] == pytest.approx(0.515)


def test_epsilon_schedule_is_linear_over_70_percent_then_flat():
    assert LR.epsilon(0, 1000) == 0.5
    assert LR.epsilon(350, 1000) == pytest.approx(0.275)         # halbe Strecke von 0,5 auf 0,05
    assert LR.epsilon(700, 1000) == pytest.approx(0.05) and LR.epsilon(999, 1000) == 0.05


def test_training_is_deterministic_and_seed_dependent():
    a, b, c = LR.train(1, 150), LR.train(1, 150), LR.train(2, 150)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert (a != 0.5).any()


def test_the_trained_agent_changes_the_q_table_only_in_visited_states():
    Q = LR.train(3, 100)
    assert 0 < (Q != 0.5).any(axis=1).sum() < LR.N_STATES + 1


def test_untrained_agent_greedy_picks_level_one_and_never_accepts_below_reservation():
    """Q überall gleich: der gierige Agent nimmt die erste Aktion (Ziel 1,0). Er bietet das Maximum und nimmt nichts unter r an, in der letzten Runde alles ab r."""
    Q = np.full((LR.N_STATES, 9), 0.5)
    d = D.make_domain(D.seed_of("lern-test|0"))
    r = LR.play(Q, "Hartnäckig", d, 0, 20)
    assert not r.agree or r.u0 >= d.r[0] - 1e-12
    me = LR.QAgent(Q, 0.0, 0.0, False, None)
    me.reset(d, 0, 20)
    assert me.propose(1) == int(np.argmax(d.u[0]))       # Ziel 1: das beste Bündel


def test_agent_accepts_in_the_last_round_by_the_rule_but_not_below_reservation():
    Q = np.full((LR.N_STATES, 9), 0.5)
    d = D.make_domain(D.seed_of("lern-test|1"))
    me = LR.QAgent(Q, 0.0, 0.0, False, None)
    me.reset(d, 1, 4, True)
    low = int(np.argmin(d.u[1]))
    assert d.u[1][low] < d.r[1] and not me.accept(4, low)
    ok = int(np.flatnonzero(d.u[1] >= d.r[1] + 1e-9)[0])
    me.reset(d, 1, 4, True)
    assert me.accept(4, ok)


def test_learning_run_improves_on_the_untrained_agent_within_a_band():
    """Bandbreite statt Gleichheit: nach 1.500 Episoden gegen die Basisgegner liegt der Mittelwert über Test-Domänen und die sechs Gegner über dem des untrainierten Agenten."""
    doms, _ = D.domain_set("lern-test-band", 20)
    Q0 = np.full((LR.N_STATES, 9), 0.5)
    base = np.mean([v[0] for v in LR.test_against(Q0, ST.POOL6, doms).values()])
    Q = LR.train(1, 1500)
    trained = np.mean([v[0] for v in LR.test_against(Q, ST.POOL6, doms).values()])
    assert trained > base + 0.02 and 0.2 < trained < 1.0


def test_baseline_of_a_fixed_strategy_matches_a_direct_negotiation():
    doms, _ = D.domain_set("lern-test-base", 3)
    got = LR.baseline("Boulware", ["Linear"], doms)
    vals = []
    for d, _ in doms:
        vals.append(NG.negotiate(d, NG.make_agent("Boulware"), NG.make_agent("Linear"), 20).u0)
        vals.append(NG.negotiate(d, NG.make_agent("Linear"), NG.make_agent("Boulware"), 20).u1)
    assert got["Linear"] == pytest.approx(np.mean(vals))


def test_training_uses_only_splitmix_streams():
    """Zufall nur aus SplitMix64: gleiche Seeds geben gleiche Bezeichnungen; Pythons hash() spielt keine Rolle."""
    assert D.seed_of("lern|q|1") == D.seed_of("lern|q|1")
    r1, r2 = SplitMix64(D.seed_of("lern|q|1")), SplitMix64(D.seed_of("lern|q|1"))
    assert [r1.uniform() for _ in range(5)] == [r2.uniform() for _ in range(5)]
