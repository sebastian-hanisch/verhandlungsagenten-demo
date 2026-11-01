"""Auswertung der Messreihen an handgerechneten Mini-Instanzen: Strategiewerte, Rangfolgen, Varianzanteile (η²), Standardabweichung mit n − 1, Positionszellen."""

import numpy as np
import pytest

import vha_results as R
import vha_tournament as TO


def _matrix(u0, u1, agree=None):
    K = len(u0)
    M = np.zeros((K, K, 7))
    for i in range(K):
        for j in range(K):
            M[i, j, TO.IX["u0"]], M[i, j, TO.IX["u1"]] = u0[i][j], u1[i][j]
            M[i, j, TO.IX["wel"]] = u0[i][j] + u1[i][j]
            M[i, j, TO.IX["agree"]] = 1.0 if agree is None else agree[i][j]
    return M


def test_strategy_scores_by_hand():
    """Drei Strategien. Nutzen der zuerst bietenden Zeile gegen Spalte (u0) und der zweiten (u1):
    u0 = [[.5,.6,.7],[.4,.5,.3],[.2,.8,.5]], u1 = [[.5,.3,.2],[.6,.5,.7],[.8,.2,.5]].
    Strategie 0: gegen 1: (u0[0][1] + u1[1][0]) / 2 = (0,6 + 0,6) / 2 = 0,6; gegen 2: (0,7 + 0,8) / 2 = 0,75 → Mittel 0,675.
    Strategie 1: gegen 0: (u0[1][0] + u1[0][1]) / 2 = (0,4 + 0,3) / 2 = 0,35; gegen 2: (0,3 + 0,2) / 2 = 0,25 → 0,30.
    Strategie 2: gegen 0: (0,2 + 0,2) / 2... u0[2][0] = 0,2, u1[0][2] = 0,2 → 0,2; gegen 1: (u0[2][1] + u1[1][2]) / 2 = (0,8 + 0,7) / 2 = 0,75 → 0,475."""
    u0 = [[.5, .6, .7], [.4, .5, .3], [.2, .8, .5]]
    u1 = [[.5, .3, .2], [.6, .5, .7], [.8, .2, .5]]
    sc = TO.strategy_scores(_matrix(u0, u1))
    assert sc["own"].tolist() == pytest.approx([0.675, 0.30, 0.475])
    assert sc["opp"][0] == pytest.approx(0.275)            # Nutzen der Gegner von Strategie 0: ((0,3 + 0,4) / 2 + (0,2 + 0,2) / 2) / 2
    members = [0, 2]
    sub = TO.strategy_scores(_matrix(u0, u1), members)
    assert sub["own"].tolist() == pytest.approx([(0.7 + 0.8) / 2, (0.2 + 0.2) / 2])           # nur noch ein Gegner: Strategie 0 gegen 2: (0,7 + 0,8) / 2; Strategie 2 gegen 0: (0,2 + 0,2) / 2


def test_ranks_and_rank_agreement():
    assert TO.ranks_desc([0.3, 0.9, 0.5]).tolist() == [3, 1, 2]
    assert TO.rank_agreement([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0) and TO.rank_agreement([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert TO.rank_agreement([3, 1, 2, 4], [4, 3, 2, 1]) == pytest.approx(np.corrcoef([2, 4, 3, 1], [1, 2, 3, 4])[0, 1])


def test_cell_stats_mean_sd_with_n_minus_one_and_top_seeds():
    u0 = [[.5, .6], [.4, .5]]
    u1 = [[.5, .3], [.6, .5]]
    a = _matrix(u0, u1)
    b = _matrix([[.5, .7], [.4, .5]], u1)
    c = R.cell_stats(np.stack([a, b]), ["A", "B"], [0, 1])
    # Strategie A: Seed 1: (0,6 + 0,6) / 2 = 0,6; Seed 2: (0,7 + 0,6) / 2 = 0,65 → Mittel 0,625, Standardabweichung mit n − 1: |0,65 − 0,6| / √2 = 0,035355
    assert c.mean["own"][0] == pytest.approx(0.625) and c.sd["own"][0] == pytest.approx(0.05 / np.sqrt(2))
    assert c.top == 0 and c.top_seeds == 2 and c.ranks.tolist() == [[1, 2], [1, 2]]


def test_eta_squared_by_hand():
    """Vier Agenten (2 Kurven × 2 Annahmen × 1 Modell) mit Werten 1, 2, 3, 4: Gesamtstreuung 5. Kurve (1,5 gegen 3,5): 4/5 = 0,8; Annahme (2 gegen 3): 1/5 = 0,2; Modell eine Stufe: 0; Rest 0."""
    combos = [("c1", "a1", "m"), ("c1", "a2", "m"), ("c2", "a1", "m"), ("c2", "a2", "m")]
    x = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    e = R.eta_squared(x, combos)
    assert e["Kurve"][0] == pytest.approx(0.8) and e["Akzeptanz"][0] == pytest.approx(0.2) and e["Modell"][0] == pytest.approx(0.0, abs=1e-12) and e["Rest"][0] == pytest.approx(0.0, abs=1e-12)
    assert e["Kurve"][1] == pytest.approx(0.0, abs=1e-12)                       # zwei gleiche Seeds: keine Streuung
    # Wechselwirkung: Werte 1, 2, 2, 1: Kurven- und Annahmemittel gleich, alles ist Rest
    e2 = R.eta_squared(np.array([[1.0, 2.0, 2.0, 1.0], [1.0, 2.0, 2.0, 1.0]]), combos)
    assert e2["Kurve"][0] == pytest.approx(0.0, abs=1e-12) and e2["Akzeptanz"][0] == pytest.approx(0.0, abs=1e-12) and e2["Rest"][0] == pytest.approx(1.0)
    # Auswahl der Stufen: ohne Annahme a2 bleiben zwei Agenten (1 und 3), nur die Kurve unterscheidet sie
    e3 = R.eta_squared(x, combos, keep=lambda cb: cb[1] != "a2")
    assert e3["Kurve"][0] == pytest.approx(1.0) and e3["Akzeptanz"][0] == pytest.approx(0.0, abs=1e-12)


def test_eta_squared_parts_add_up_to_one_on_the_real_48_agents():
    data = R.XP.load()
    arrs, combos = R.decomp_arrays(data), R.decomp_combos()
    assert len(combos) == 48 and arrs["own"].shape == (4, 48)
    for m in R.DECOMP_METRICS:
        e = R.eta_squared(arrs[m], combos)
        assert sum(v[0] for v in e.values()) == pytest.approx(1.0, abs=1e-9)


def test_last_bidder_by_parity():
    assert [R.last_bidder(T) for T in (2, 3, 6, 7, 20, 21)] == [1, 0, 1, 0, 1, 0]


def test_ms_uses_n_minus_one():
    assert R._ms([1.0, 3.0]) == (2.0, pytest.approx(np.sqrt(2.0)))
