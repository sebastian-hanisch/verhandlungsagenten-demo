"""Ablauf einer Aushandlung: abwechselnde Angebote, der Antwortende nimmt an oder lehnt ab, Frist T Runden, ohne Einigung bekommt jede Seite ihren Reservationsnutzen.

Position 0 bietet in den Runden 1, 3, 5, …, Position 1 in 2, 4, 6, …; bei geradem T bietet also Position 1 **zuletzt**, bei ungeradem T Position 0.
**Regel „letzte Runde“** (`last_round_rule`): in Runde T nimmt der Antwortende jedes Angebot ab seinem Reservationsnutzen an (eine mehrdeutige Rechenregel, in der App wählbar).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import vha_opponent as OPP
import vha_strategies as ST
from vha_domain import EPS, Domain

BAND = 0.05      # Standardbreite des Fensters um das Ziel (Isolinie)


class Result(NamedTuple):
    agree: bool
    t: int            # Runde der Einigung, sonst T
    offer: int        # Bündel der Einigung, sonst -1
    u0: float         # Nutzen der Position 0 (Reservationsnutzen ohne Einigung)
    u1: float


class Agent:
    """Ein Verhandlungsprogramm mit seinem Zustand in einer Aushandlung (beobachtete Angebote, Tit-for-Tat-Ziel, Häufigkeitsmodell)."""

    def __init__(self, spec: ST.Spec, name: str = ""):
        self.spec, self.name = spec, name

    # -- Vorbereitung
    def reset(self, dom: Domain, pos: int, horizon: int, last_round_rule: bool = True, band: float = BAND):
        self.dom, self.pos, self.T, self.L, self.band = dom, pos, horizon, last_round_rule, band
        self.u = dom.u[pos]
        self.uo = dom.u[1 - pos]
        self.r = float(dom.r[pos])
        self.opp: list[int] = []
        self.tft = 1.0
        self._cache: dict[int, int] = {}
        self.counts, self.nsame = OPP.new_state(dom.n, dom.m)

    # -- Beobachtung
    def observe(self, offer: int):
        """Ein Angebot des Gegners zur Kenntnis nehmen (Modell und Tit-for-Tat-Ziel nachführen)."""
        if self.opp:
            prev = self.opp[-1]
            self.tft = ST.tft_target(self.tft, float(self.u[offer] - self.u[prev]), self.spec.gamma, self.r)
        OPP.update_counts(self.counts, self.nsame, self.dom.B[offer], self.dom.B[self.opp[-1]] if self.opp else None)
        self.opp.append(offer)
        self._cache = {}

    # -- Gegnermodell
    def est(self) -> np.ndarray | None:
        if self.spec.model == "keins":
            return None
        if self.spec.model == "orakel":
            return self.uo
        return OPP.estimate_utility(self.counts, self.nsame, self.dom.B)

    def est_weights(self) -> np.ndarray:
        return OPP.estimate_weights(self.nsame)

    # -- Gebot
    def target(self, t: int) -> float:
        c = self.spec.curve
        if c == "hart":
            return 1.0
        if c == "tft":
            return self.tft
        return ST.target_curve(t, self.T, self.r, self.spec.beta)

    def choose(self, target: float) -> int:
        return ST.window_choice(self.u, target, self.band, self.est(), self.dom.tie)

    def bid(self, t: int) -> int:
        if t in self._cache:
            return self._cache[t]
        b = None
        if self.spec.endgame and t == self.T and self.est() is not None:
            b = ST.endgame_choice(self.u, self.est(), float(self.dom.r[1 - self.pos]), self.dom.tie)
        if b is None:
            b = self.choose(self.target(t))
        self._cache[t] = b
        return b

    propose = bid

    # -- Annahme
    def accept(self, t: int, offer: int) -> bool:
        u = float(self.u[offer])
        if self.spec.ac == "never":
            return False
        if u < self.r - EPS:
            return False
        if self.L and t == self.T:
            return True
        u_next = float(self.u[self.bid(t + 1)]) if self.spec.ac == "next" else 0.0
        return ST.accept_rule(self.spec.ac, u, t, self.T, self.r, self.spec.theta, u_next)

    def finish(self, final_u: float):
        pass


def make_agent(spec_or_name) -> Agent:
    if isinstance(spec_or_name, str):
        return Agent(ST.spec_of(spec_or_name), spec_or_name)
    return Agent(spec_or_name)


def negotiate(dom: Domain, a0, a1, horizon: int, last_round_rule: bool = True, band: float = BAND, trace: list | None = None) -> Result:
    """Eine Aushandlung. `a0` (Position 0) bietet zuerst. `trace` (Liste) bekommt je Runde `(Runde, bietende Position, Bündel, angenommen, Ziel des Bietenden)`;
    das Ziel ist der Zielnutzen vor dem Gebot (das letzte Gebot eines Endspiel-Programms weicht davon ab)."""
    agents = (a0, a1)
    a0.reset(dom, 0, horizon, last_round_rule, band)
    a1.reset(dom, 1, horizon, last_round_rule, band)
    res = None
    for t in range(1, horizon + 1):
        p = (t - 1) % 2
        q = 1 - p
        goal = agents[p].target(t) if trace is not None and hasattr(agents[p], "target") else float("nan")
        o = agents[p].propose(t)
        agents[q].observe(o)
        acc = agents[q].accept(t, o)
        if trace is not None:
            trace.append((t, p, o, bool(acc), float(goal)))
        if acc:
            res = Result(True, t, o, float(dom.u[0][o]), float(dom.u[1][o]))
            break
    if res is None:
        res = Result(False, horizon, -1, float(dom.r[0]), float(dom.r[1]))
    a0.finish(res.u0)
    a1.finish(res.u1)
    return res


def evaluate(dom: Domain, info: dict, res: Result) -> dict:
    """Kennzahlen einer Aushandlung: Einigung, Runde, Nutzen beider Seiten, Summe, **Abstand zur Pareto-Grenze** (kleinster Abstand zu einem Punkt der Grenze der individuell rationalen Bündel)
    und das **Nash-Produkt relativ** zu dem der Nash-Lösung (0 ohne Einigung)."""
    F = info["F"]
    dist = float(np.sqrt(((F - np.array([res.u0, res.u1])) ** 2).sum(1)).min())
    nr = ((res.u0 - dom.r[0]) * (res.u1 - dom.r[1]) / info["nashprod"]) if res.agree and info["nashprod"] > 0 else 0.0
    return dict(agree=res.agree, t=res.t, u0=res.u0, u1=res.u1, wel=res.u0 + res.u1, dist=dist, nashrel=float(nr))
