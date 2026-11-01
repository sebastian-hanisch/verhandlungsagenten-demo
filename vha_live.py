"""Live-Rechnungen der App, die nicht vorgerechnet sind: ein Pool-Turnier mit wenigen Domänen und ein kurzes Training des Lernenden. Beide laufen erst nach einem Klick."""

from __future__ import annotations

import numpy as np

import vha_constants as C
import vha_domain as D
import vha_learning as LR
import vha_strategies as ST
import vha_tournament as TO
from vha_scenario import Settings


def live_domains(st: Settings, count: int, tag: str) -> list:
    """`count` gültige Domänen mit den Domäneneinstellungen der Seitenleiste; die Seeds folgen aus dem Zufalls-Seed der Seitenleiste."""
    doms, _ = D.domain_set(f"live|{tag}|{st.seed}", count, n=st.n, m=st.m, s=st.s, g=st.g, r=st.r)
    return doms


def live_tournament(st: Settings, pool: str, count: int) -> dict:
    """Pool-Turnier (jeder gegen jeden, beide Positionen) über `count` Domänen mit Frist, Regel und Fenster der Seitenleiste. `pool`: „6“ (Standard) oder „7“ (mit Endspiel-Ausnutzer)."""
    names = list(ST.POOL6 if pool == "6" else ST.POOL7)
    doms = live_domains(st, count, "turnier")
    M = TO.pair_means(names, doms, st.T, st.rule_on, st.band)
    sc = TO.strategy_scores(M)
    return {"names": names, "M": M, "scores": sc, "count": len(doms), "negotiations": len(names) ** 2 * len(doms)}


def live_learner(st: Settings, episodes: int = C.LIVE_EPISODES, test_count: int = 30) -> dict:
    """Training des Lernenden in der Standarddomäne (4 Punkte × 4 Werte, s = 0,5, g = 0,5, r = 0,2, Frist 20, Regel an) gegen die fünf Basisgegner, Test in der Domäne und mit den Einstellungen der Seitenleiste
    (Frist, Regel, Fenster) gegen alle sechs Pool-Strategien; zum Vergleich Boulware + Modell auf denselben Test-Domänen."""
    Q = LR.train(st.seed, episodes, LR.BASE5, 20)
    doms = live_domains(st, test_count, "lerner")
    test = LR.test_against(Q, ST.POOL6, doms, st.T, st.rule_on, st.band)
    base = LR.baseline(ST.BOULWARE_MODELL, ST.POOL6, doms, st.T, st.rule_on, st.band)
    return {"episodes": episodes, "test": test, "base": base, "mean": float(np.mean([v[0] for v in test.values()])), "base_mean": float(np.mean(list(base.values()))), "count": len(doms)}
