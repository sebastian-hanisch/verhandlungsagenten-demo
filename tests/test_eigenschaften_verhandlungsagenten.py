"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, Spieltheorie und Simulation): Aussagen, die für die Aushandlung gelten müssen, ohne dass man das richtige Ergebnis kennt.

Umnummerieren der Punkte und Werte ändert kein Ergebnis; derselbe Seed gibt dasselbe Ergebnis; in einer gleichgerichteten Domäne einigen sich alle Paare in Runde 1; Rollentausch im Selbstspiel
(statistisch); Ergebnisse liegen zwischen Reservationsnutzen und 1; Nutzenpaare und Kennzahlen sind konsistent; die Strategiewerte hängen nicht an der Reihenfolge der Strategien im Pool.
Eine affine Umrechnung der Nutzen ist hier **keine** Eigenschaft: die Zielkurven setzen den größten Nutzen 1 voraus."""

import itertools

import numpy as np
import pytest

import vha_domain as D
import vha_negotiation as NG
import vha_strategies as ST
import vha_tournament as TO


def test_renumbering_points_and_values_changes_no_result():
    """Eigenschaft: Punkte und Werte umnummerieren (mit festem Zufallsvektor je Bündel, mit umgeordnet) ändert kein Ergebnis einer Aushandlung."""
    mism = tot = 0
    for k in range(8):
        rng = np.random.default_rng([21, k])
        d = D.make_domain(D.seed_of(f"renum|{k}"), n=4, m=3, s=0.6, g=0.5)
        n, m = d.n, d.m
        pi = rng.permutation(n)
        sig = [rng.permutation(m) for _ in range(n)]
        w2 = d.w[:, pi]
        v2 = [np.stack([d.v[p][pi[j]][sig[j]] for j in range(n)]) for p in (0, 1)]
        old_idx = np.zeros(d.N, dtype=int)
        for idx_new, x in enumerate(D.bundles(n, m)):
            ob = np.zeros(n, dtype=int)
            for j in range(n):
                ob[pi[j]] = sig[j][x[j]]
            oi = 0
            for i in range(n):
                oi = oi * m + ob[i]
            old_idx[idx_new] = oi
        d2 = D.Domain(w2, v2, d.r, d.tie[old_idx])
        assert np.allclose(d2.u[0], d.u[0][old_idx]) and np.allclose(d2.u[1], d.u[1][old_idx])
        for a, b in itertools.product(ST.POOL6, repeat=2):
            r1 = NG.negotiate(d, NG.make_agent(a), NG.make_agent(b), 20)
            r2 = NG.negotiate(d2, NG.make_agent(a), NG.make_agent(b), 20)
            tot += 1
            mism += not (r1.agree == r2.agree and r1.t == r2.t and abs(r1.u0 - r2.u0) < 1e-9 and abs(r1.u1 - r2.u1) < 1e-9)
    assert tot == 8 * 36 and mism == 0


def test_same_domain_same_strategies_same_result():
    d = D.make_domain(D.seed_of("det"))
    a = [NG.negotiate(d, NG.make_agent("Boulware + Modell"), NG.make_agent("Tit-for-Tat (relativ)"), 20) for _ in range(2)]
    assert a[0] == a[1]


def test_fully_aligned_domain_every_pair_agrees_in_round_one_on_utility_one():
    """u₁ = u₀: jede Seite bekommt im ersten Angebot (Ziel 1) ihr Maximum, also nimmt die zweite sofort an (ihr nächstes Gebot bringt höchstens 1). Alle 36 Paare × 6 Domänen."""
    bad = 0
    for k in range(6):
        d = D.make_domain(D.seed_of(f"aligned|{k}"), n=4, m=4, s=0.0, g=0.0)
        same = D.Domain([d.w[0], d.w[0]], [d.v[0], d.v[0]], d.r, d.tie)
        for a, b in itertools.product(ST.POOL6, repeat=2):
            res = NG.negotiate(same, NG.make_agent(a), NG.make_agent(b), 20)
            bad += not (res.agree and res.t == 1 and abs(res.u0 - 1) < 1e-9 and abs(res.u1 - 1) < 1e-9)
    assert bad == 0


@pytest.mark.parametrize("name", ST.POOL6)
def test_role_swap_in_self_play_gives_both_sides_the_same_mean_within_the_standard_error(name):
    """Rollentausch: Seite A bietet zuerst in D, Seite B in D mit vertauschten Seiten. Im Selbstspiel gleicher Strategien ist der Mittelunterschied von A und B
    höchstens 3,5 Standardfehler (statistisch, nicht exakt: Positionen sind nicht symmetrisch, aber die Seiten der Zufallsdomänen sind es)."""
    a_vals, b_vals = [], []
    for k in range(150):
        d = D.make_domain(D.seed_of(f"swap|{k}"))
        r1 = NG.negotiate(d, NG.make_agent(name), NG.make_agent(name), 20)
        r2 = NG.negotiate(D.swap_sides(d), NG.make_agent(name), NG.make_agent(name), 20)
        a_vals.append((r1.u0 + r2.u1) / 2)
        b_vals.append((r1.u1 + r2.u0) / 2)
    diff = np.array(a_vals) - np.array(b_vals)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    assert abs(diff.mean()) < 3.5 * max(se, 1e-9)


def test_agreement_utilities_never_below_the_reservation_and_in_the_unit_interval():
    for k in range(10):
        d = D.make_domain(D.seed_of(f"bounds|{k}"))
        for a, b in itertools.product(ST.POOL6, repeat=2):
            res = NG.negotiate(d, NG.make_agent(a), NG.make_agent(b), 12)
            assert res.u0 >= d.r[0] - 1e-12 and res.u1 >= d.r[1] - 1e-12 and res.u0 <= 1 + 1e-12 and res.u1 <= 1 + 1e-12
            assert 1 <= res.t <= 12 and (res.offer >= 0) == res.agree


def test_evaluate_is_consistent_for_all_pairs():
    """Summe = Nutzen von Position 0 + 1, Abstand zur Grenze ≥ 0, relatives Nash-Produkt höchstens 1 (die Nash-Lösung maximiert unter den rationalen Bündeln), 0 ohne Einigung."""
    doms, _ = D.domain_set("eigenschaften|eval", 8)
    for d, info in doms:
        for a, b in itertools.product(ST.POOL6, repeat=2):
            res = NG.negotiate(d, NG.make_agent(a), NG.make_agent(b), 20)
            e = NG.evaluate(d, info, res)
            assert e["wel"] == pytest.approx(res.u0 + res.u1) and e["dist"] >= 0.0
            assert 0.0 <= e["nashrel"] <= 1.0 + 1e-9 and (e["nashrel"] == 0.0 or e["agree"])
            if res.agree:
                assert info["ir"][res.offer]
            else:
                assert (res.u0, res.u1) == (d.r[0], d.r[1])


def test_strategy_scores_do_not_depend_on_the_order_of_the_pool():
    doms, _ = D.domain_set("eigenschaften|order", 6)
    names = list(ST.POOL6)
    M = TO.pair_means(names, doms, 12)
    perm = [3, 0, 5, 1, 4, 2]
    M2 = TO.pair_means([names[i] for i in perm], doms, 12)
    s1, s2 = TO.strategy_scores(M), TO.strategy_scores(M2)
    for k in TO.SCORE_KEYS:
        assert s2[k] == pytest.approx(s1[k][perm])


def test_a_longer_horizon_never_lowers_the_agreement_rate_of_the_boulware_pair_on_average():
    """Mehr Zeit gibt Boulware gegen Boulware mehr Gelegenheit zur Einigung (Mittel über 40 Domänen, Regel an): Einigungsrate bei T = 40 mindestens so groß wie bei T = 6."""
    doms, _ = D.domain_set("eigenschaften|horizon", 40)
    rate = lambda T: np.mean([NG.negotiate(d, NG.make_agent("Boulware"), NG.make_agent("Boulware"), T).agree for d, _ in doms])   # noqa: E731
    assert rate(40) >= rate(6)
