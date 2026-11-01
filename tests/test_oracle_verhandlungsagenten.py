"""Orakel (ORACLE-PLAYBOOK §2, Spieltheorie): Pareto-Grenze und Nash-Lösung gegen das Aufzählen aller Paare in reinem Python und gegen Maximierer gewichteter Summen;
der Spezialfall m = 2, g = 1 gegen die Reihenfolge-Regel von Stück G
(unabhängig hier neu geschrieben: Punkte mit dem größten w₁ / w₀ gehen zuerst an die zweite Seite). Eigenschaften der Aushandlung stehen in test_eigenschaften_verhandlungsagenten.py.

Das Orakel ist vorher an der Mini-Domäne von Hand geprüft (`tests/test_domain.py`: Grenze {0, 1, 2, 5, 8}, rationale Grenze {1, 2, 5}, Nash b2 mit Produkt 0,20)."""


import numpy as np
import pytest

import vha_domain as D


def brute_front(pts, ids):
    """O(N²): i ist dominiert, wenn ein j in `ids` in beiden Nutzen mindestens so gut und in einem besser ist."""
    out = []
    for i in ids:
        dominated = False
        for j in ids:
            if pts[j][0] >= pts[i][0] and pts[j][1] >= pts[i][1] and (pts[j][0] > pts[i][0] or pts[j][1] > pts[i][1]):
                dominated = True
                break
        if not dominated:
            out.append(i)
    return out


def _rounded_points(d):
    return [(round(float(a), 12), round(float(b), 12)) for a, b in zip(d.u[0], d.u[1])]


def test_front_and_nash_against_enumeration_in_100_random_domains():
    checked = 0
    for k in range(100):
        n, m = [(3, 3), (3, 4), (4, 3), (2, 5)][k % 4]
        rng = np.random.default_rng([11, k])
        d = D.make_domain(D.seed_of(f"orakel|{k}"), n=n, m=m, s=float(rng.random()), g=float(rng.random()), r=0.2 * float(rng.random()))
        info = D.dom_info(d)
        if info is None:
            continue
        checked += 1
        pts = _rounded_points(d)
        ir = [i for i, (a, b) in enumerate(pts) if a >= d.r[0] - 1e-12 and b >= d.r[1] - 1e-12]
        assert sorted(brute_front(pts, ir)) == info["front"].tolist()
        best = max((pts[i][0] - d.r[0]) * (pts[i][1] - d.r[1]) for i in ir)
        assert info["nashprod"] == pytest.approx(best, abs=1e-12)
        assert pts[info["nash"]][0] - d.r[0] >= -1e-12
        assert info["maxsum"] == pytest.approx(max(pts[i][0] + pts[i][1] for i in ir), abs=1e-12)
        assert info["nir"] == len(ir)
    assert checked >= 90


def test_weighted_sum_maximisers_choosing_per_point_lie_on_the_frontier_of_all_bundles():
    """Jede positive Gewichtung λ·u₀ + (1 − λ)·u₁ wird bündelweise je Punkt getrennt maximiert (additive Nutzen): das Ergebnis liegt auf der Pareto-Grenze aller Bündel."""
    bad = total = 0
    for k in range(60):
        n, m = [(3, 3), (3, 4), (4, 3), (2, 5)][k % 4]
        d = D.make_domain(D.seed_of(f"orakel-ws|{k}"), n=n, m=m, s=0.7, g=0.4, r=0.0)
        front = set(D.pareto_idx(d.u[0], d.u[1]).tolist())
        for lam in (0.1, 0.3, 0.5, 0.7, 0.9):
            wv = lam * d.w[0][:, None] * d.v[0] + (1 - lam) * d.w[1][:, None] * d.v[1]
            idx = 0
            for i in range(n):
                idx = idx * m + int(np.argmax(wv[i]))
            total += 1
            bad += idx not in front
    assert total == 300 and bad == 0


def test_every_front_point_is_a_weighted_sum_maximiser_or_non_supported_but_never_dominated():
    """Gegenprobe zur Grenze selbst: kein Bündel der Grenze wird von einem anderen Bündel dominiert, und jedes andere Bündel wird von einem der Grenze (schwach) dominiert."""
    d = D.make_domain(D.seed_of("orakel-dom"), n=3, m=4, s=0.5, g=0.5)
    front = D.pareto_idx(d.u[0], d.u[1]).tolist()
    for i in range(d.N):
        dominated_by_front = any(d.u[0][j] >= d.u[0][i] - 1e-12 and d.u[1][j] >= d.u[1][i] - 1e-12 for j in front)
        assert dominated_by_front
    for i in front:
        assert not any(d.u[0][j] > d.u[0][i] + 1e-12 and d.u[1][j] >= d.u[1][i] - 1e-12 or d.u[1][j] > d.u[1][i] + 1e-12 and d.u[0][j] >= d.u[0][i] - 1e-12 for j in range(d.N))


def test_special_case_binary_points_and_opposite_values_is_the_order_rule_of_piece_g():
    """m = 2, g = 1, r = 0: die Ecken der Reihenfolge „Punkte mit dem größten w₁ / w₀ gehen zuerst an die zweite Seite“ liegen alle auf der Pareto-Grenze aller Bündel."""
    bad = total = 0
    for k in range(60):
        d = D.make_domain(D.seed_of(f"binaer|{k}"), n=5, m=2, s=1.0, g=1.0, r=0.0)
        allf = D.pareto_idx(d.u[0], d.u[1])
        pts = {(round(float(d.u[0][i]), 9), round(float(d.u[1][i]), 9)) for i in allf}
        order = np.argsort(-(d.w[1] / d.w[0]))
        to1: set[int] = set()
        for step in range(d.n + 1):
            u0 = sum(d.w[0][i] for i in range(d.n) if i not in to1)
            u1 = sum(d.w[1][i] for i in to1)
            total += 1
            bad += (round(u0, 9), round(u1, 9)) not in pts
            if step < d.n:
                to1.add(int(order[step]))
    assert total == 60 * 6 and bad == 0
