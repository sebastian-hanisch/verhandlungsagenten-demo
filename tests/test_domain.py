"""Domäne gegen Handrechnung: Nutzen, Pareto-Grenze, Nash-Lösung der Mini-Domäne; Erzeugung der Zufallsdomänen (Normierung, Gewichte, Determinismus)."""

import numpy as np
import pytest

import vha_domain as D
from conftest import MINI_U0, MINI_U1


def test_bundles_are_all_value_combinations_in_product_order():
    b = D.bundles(2, 3)
    assert b.shape == (9, 2) and b.tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    assert D.bundles(3, 4).shape == (64, 3) and D.bundles(5, 5).shape == (3125, 5)
    assert not D.bundles(2, 3).flags.writeable


def test_mini_utilities_match_the_hand_calculation(mini):
    assert np.allclose(mini.u[0], MINI_U0, atol=1e-12) and np.allclose(mini.u[1], MINI_U1, atol=1e-12)


def test_mini_pareto_frontier_of_all_bundles_is_0_1_2_5_8(mini):
    """Von Hand: b0 (1; 0), b1 (0,8; 0,35), b2 (0,6; 0,7), b5 (0,3; 0,85), b8 (0; 1) bilden die Grenze; b3 (0,7; 0,15) wird von b1 dominiert,
    b4 (0,5; 0,5) von b2, b6 (0,4; 0,3) von b2, b7 (0,2; 0,65) von b2."""
    assert D.pareto_idx(mini.u[0], mini.u[1]).tolist() == [0, 1, 2, 5, 8]


def test_mini_info_rational_bundles_frontier_and_nash(mini):
    """Individuell rational (beide ≥ 0,2): b1, b2, b4, b5, b6, b7 (b0 gibt Position 1 nichts, b3 0,15, b8 gibt Position 0 nichts).
    Grenze unter ihnen: b1, b2, b5. Nash-Produkte (u₀ − 0,2)(u₁ − 0,2): b1 0,6·0,15 = 0,09; b2 0,4·0,5 = 0,20; b5 0,1·0,65 = 0,065; b4 0,3·0,3 = 0,09 → Nash b2 mit 0,20. Größte Summe: b2 1,3."""
    info = D.dom_info(mini)
    assert np.flatnonzero(info["ir"]).tolist() == [1, 2, 4, 5, 6, 7]
    assert info["front"].tolist() == [1, 2, 5]
    assert info["nash"] == 2 and info["nashprod"] == pytest.approx(0.20, abs=1e-12)
    assert info["maxsum"] == pytest.approx(1.3, abs=1e-12) and info["nir"] == 6


def test_dom_info_is_none_when_no_bundle_is_rational(mini):
    hard = D.Domain(mini.w, mini.v, [0.95, 0.95], mini.tie)
    assert D.dom_info(hard) is None


def test_equal_utility_pairs_are_all_kept_on_the_frontier():
    u0 = np.array([0.5, 0.5, 0.2, 0.9])
    u1 = np.array([0.5, 0.5, 0.2, 0.1])
    assert D.pareto_idx(u0, u1).tolist() == [0, 1, 3]


def test_swap_sides_swaps_utilities_and_reservations(mini):
    sw = D.swap_sides(D.Domain(mini.w, mini.v, [0.1, 0.3], mini.tie))
    assert np.allclose(sw.u[0], MINI_U1) and np.allclose(sw.u[1], MINI_U0) and sw.r.tolist() == [0.3, 0.1]


@pytest.mark.parametrize("n,m", [(3, 3), (4, 4), (5, 5), (2, 2)])
def test_random_domain_is_normalised(n, m):
    for k in range(10):
        d = D.make_domain(D.seed_of(f"t|{n}|{m}|{k}"), n=n, m=m, s=0.5, g=0.5)
        assert d.u[0].max() == pytest.approx(1.0) and d.u[1].max() == pytest.approx(1.0)
        assert d.u[0].min() == pytest.approx(0.0, abs=1e-12) and d.u[1].min() == pytest.approx(0.0, abs=1e-12)
        assert d.w.sum(1) == pytest.approx([1.0, 1.0]) and (d.w > 0).all()
        assert all((v.min(1) == 0).all() and (v.max(1) == 1).all() for v in d.v)


def test_s_zero_gives_equal_weights_and_s_one_gives_a_large_gap():
    gaps0 = [D.weight_gap(D.make_domain(k, s=0.0)) for k in range(50)]
    gaps1 = [D.weight_gap(D.make_domain(k, s=1.0)) for k in range(50)]
    assert max(gaps0) < 1e-12 and np.mean(gaps1) > 0.15


def test_g_one_makes_the_value_lists_exactly_opposite_and_g_zero_does_not():
    d = D.make_domain(7, g=1.0)
    assert np.allclose(d.v[1], 1.0 - d.v[0])
    d0 = D.make_domain(7, g=0.0)
    assert not np.allclose(d0.v[1], 1.0 - d0.v[0])


def test_same_seed_same_domain_and_different_seed_different_domain():
    a, b, c = D.make_domain(5), D.make_domain(5), D.make_domain(6)
    assert np.array_equal(a.u[0], b.u[0]) and np.array_equal(a.tie, b.tie) and not np.array_equal(a.u[0], c.u[0])


def test_seed_of_is_fnv1a_and_stable():
    assert D.seed_of("") == 0xCBF29CE484222325 and D.seed_of("a") == 0xAF63DC4C8601EC8C
    assert D.seed_of("x") != D.seed_of("y")


def test_domain_set_skips_invalid_domains_and_counts_them():
    doms, skipped = D.domain_set("test-set", 20, n=3, m=3, r=0.45)
    assert len(doms) == 20 and skipped >= 0 and all(info is not None for _, info in doms)
    doms2, skipped2 = D.domain_set("test-set", 20, n=3, m=3, r=0.45)
    assert skipped == skipped2 and [d.u[0].tolist() for d, _ in doms[:3]] == [d.u[0].tolist() for d, _ in doms2[:3]]
