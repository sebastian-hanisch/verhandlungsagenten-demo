"""Lernender Verhandler: tabellarisches Q-Learning auf einem stark verdichteten Zustand (in der Art von RLBOA, neu und klein geschrieben, keine Kopie).

**Zustand** (90): Zeitfenster (5) × Nutzen des letzten Angebots des Gegners für mich (5 Klassen oder „noch keins“ = 6) × Trend gegenüber dem vorigen Angebot (3: fällt, gleich, steigt).
**Aktion** (9): Zielnutzen aus {1; 0,9; …; 0,2}, nach unten durch den Reservationsnutzen begrenzt. Angenommen wird ein Angebot ab diesem Ziel; sonst bietet der Agent auf der Isolinie des
Ziels (Fenster wie bei den festen Strategien) das Bündel mit dem größten Gegnernutzen laut Häufigkeitsmodell. **Belohnung:** der Nutzen am Ende (Reservationsnutzen ohne Einigung), kein Diskont.
**Training:** frische Zufallsdomäne je Episode, Gegner und Position zufällig aus den Trainingsgegnern; Lernrate 0,1; ε sinkt linear von 0,5 auf 0,05 über die ersten 70 % der Episoden.
Zufall nur über SplitMix64. Modellannahmen: kein Ergebnis gilt über den Trainingspool und diesen Domänengenerator hinaus.
"""

from __future__ import annotations

import numpy as np

import vha_domain as D
import vha_negotiation as NG
import vha_strategies as ST
from vha_rng import SplitMix64

LEVELS = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
N_TIME, N_UTIL, N_TREND = 5, 6, 3
N_STATES = N_TIME * N_UTIL * N_TREND          # 90
ALPHA = 0.1
EPS_START, EPS_END, EPS_SHARE = 0.5, 0.05, 0.7
TRAIN_S = 0.5                                  # Gewichtsunterschied der Trainingsdomänen
BASE5 = (ST.HARTNAECKIG, ST.BOULWARE, ST.LINEAR, ST.CONCEDER, ST.TFT)       # Trainingsgegner (die sechste Strategie des Pools kommt im Training nicht vor)
LEARNER_SPEC = ST.Spec(curve="hart", ac="q", model="haeufig")


def state_index(t: int, horizon: int, u_offer: float | None, u_prev: float | None) -> int:
    """Zustand aus Runde, Nutzen des letzten Gegnerangebots (Klasse ⌊5u⌋, höchstens 4; „kein Angebot“ = 5) und Trend (±0,02 Totband)."""
    tb = min(N_TIME - 1, int(N_TIME * (t - 1) / (horizon - 1)))
    if u_offer is None:
        return (tb * N_UTIL + 5) * N_TREND + 1
    ub = min(4, int(u_offer * 5))
    if u_prev is None:
        tr = 1
    else:
        dlt = u_offer - u_prev
        tr = 2 if dlt > 0.02 else (0 if dlt < -0.02 else 1)
    return (tb * N_UTIL + ub) * N_TREND + tr


def td_update(Q: np.ndarray, s_prev: int, a_prev: int, target: float, alpha: float = ALPHA) -> None:
    """Q(s, a) ← Q(s, a) + α · (Ziel − Q(s, a)); das Ziel ist max_a' Q(s', a') im Zwischenschritt und der Nutzen am Ende im letzten Schritt (kein Diskont)."""
    Q[s_prev, a_prev] += alpha * (target - Q[s_prev, a_prev])


def epsilon(episode: int, total: int) -> float:
    return max(EPS_END, EPS_START - (EPS_START - EPS_END) * episode / (EPS_SHARE * total))


class QAgent(NG.Agent):
    def __init__(self, Q: np.ndarray, eps: float, alpha: float, learn: bool, rng: SplitMix64 | None):
        super().__init__(LEARNER_SPEC, "Lernender")
        self.Q, self.eps, self.alpha, self.learn, self.rng = Q, eps, alpha, learn, rng

    def reset(self, dom, pos, horizon, last_round_rule=True, band=NG.BAND):
        super().reset(dom, pos, horizon, last_round_rule, band)
        self.last = None
        self.pend_a = None
        self.prev_u = None

    def decide(self, t: int, offer: int | None) -> int:
        u = None if offer is None else float(self.u[offer])
        s = state_index(t, self.T, u, self.prev_u)
        if self.learn and self.rng.uniform() < self.eps:
            a = int(self.rng.uniform() * len(LEVELS))
        else:
            q = self.Q[s]
            best = np.flatnonzero(q >= q.max() - 1e-12)
            a = int(best[int(self.rng.uniform() * len(best))]) if self.learn else int(best[0])
        if self.learn and self.last is not None:
            td_update(self.Q, self.last[0], self.last[1], float(self.Q[s].max()), self.alpha)
        self.last = (s, a)
        if u is not None:
            self.prev_u = u
        return a

    def accept(self, t: int, offer: int) -> bool:
        a = self.decide(t, offer)
        self.pend_a = a
        u = float(self.u[offer])
        if u < self.r - NG.EPS:
            return False
        if self.L and t == self.T:
            return True
        return u >= max(self.r, LEVELS[a]) - NG.EPS

    def propose(self, t: int) -> int:
        if self.pend_a is None:
            a = self.decide(t, None)
        else:
            a = self.pend_a
            self.pend_a = None
        return self.choose(max(self.r, float(LEVELS[a])))

    bid = propose

    def finish(self, final_u: float):
        if self.learn and self.last is not None:
            td_update(self.Q, self.last[0], self.last[1], final_u, self.alpha)


def train(seed: int, episodes: int, pool=BASE5, horizon: int = 20, rule: bool = True, band: float = NG.BAND) -> np.ndarray:
    """Q-Tabelle nach `episodes` Episoden gegen die Gegner aus `pool` (ε-Plan über die ganze Länge)."""
    Q = np.full((N_STATES, len(LEVELS)), 0.5)
    pick = SplitMix64(D.seed_of(f"lern|gegner|{seed}|{len(pool)}"))
    qrng = SplitMix64(D.seed_of(f"lern|q|{seed}"))
    for ep in range(episodes):
        d = D.make_domain(D.seed_of(f"lern|dom|{seed}|{ep}"), s=TRAIN_S)
        opp = NG.make_agent(pool[int(pick.uniform() * len(pool))])
        me = QAgent(Q, epsilon(ep, episodes), ALPHA, True, qrng)
        if pick.uniform() < 0.5:
            NG.negotiate(d, me, opp, horizon, rule, band)
        else:
            NG.negotiate(d, opp, me, horizon, rule, band)
    return Q


def play(Q: np.ndarray, opponent, dom: D.Domain, pos: int, horizon: int, rule: bool = True, band: float = NG.BAND) -> NG.Result:
    """Eine Aushandlung des (nicht mehr lernenden, gierigen) Agenten gegen `opponent` (Name oder Spec) in Position `pos`."""
    me = QAgent(Q, 0.0, 0.0, False, None)
    opp = NG.make_agent(opponent)
    return NG.negotiate(dom, me, opp, horizon, rule, band) if pos == 0 else NG.negotiate(dom, opp, me, horizon, rule, band)


def test_against(Q: np.ndarray, opponents, doms, horizon: int = 20, rule: bool = True, band: float = NG.BAND) -> dict[str, tuple[float, float]]:
    """Je Gegner (eigener Nutzen, Einigungsrate), gemittelt über die Domänen und beide Positionen."""
    out = {}
    for on in opponents:
        vals, ag = [], []
        for d, _ in doms:
            for pos in (0, 1):
                r = play(Q, on, d, pos, horizon, rule, band)
                vals.append(r.u0 if pos == 0 else r.u1)
                ag.append(1.0 if r.agree else 0.0)
        out[on] = (float(np.mean(vals)), float(np.mean(ag)))
    return out


def baseline(spec_name: str, opponents, doms, horizon: int = 20, rule: bool = True, band: float = NG.BAND) -> dict[str, float]:
    """Eigener Nutzen einer festen Strategie gegen dieselben Gegner und Domänen wie `test_against`."""
    out = {}
    for on in opponents:
        vals = []
        for d, _ in doms:
            for pos in (0, 1):
                me, op = NG.make_agent(spec_name), NG.make_agent(on)
                r = NG.negotiate(d, me, op, horizon, rule, band) if pos == 0 else NG.negotiate(d, op, me, horizon, rule, band)
                vals.append(r.u0 if pos == 0 else r.u1)
        out[on] = float(np.mean(vals))
    return out
