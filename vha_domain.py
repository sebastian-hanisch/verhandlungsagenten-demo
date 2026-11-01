"""Domäne der Verhandlung: n Punkte mit je m Werten, linear-additive Nutzen, Pareto-Grenze und Nash-Lösung durch Aufzählen aller Bündel.

Ein **Bündel** wählt für jeden Punkt einen Wert; es gibt `m ** n` Bündel (Reihenfolge wie `itertools.product`, der letzte Punkt läuft am schnellsten).
Der Nutzen einer Seite p ist `u_p(b) = Σ_i w[p][i] · v[p][i][b_i]` mit Gewichten `w[p]` (Summe 1) und Wertlisten `v[p][i]`, je Punkt auf Minimum 0 und Maximum 1 normiert:
der größte Nutzen einer Seite ist 1 (die Punkte sind unabhängig, also lassen sich die besten Werte zu einem Bündel verbinden).
Ohne Einigung bekommt jede Seite ihren Reservationsnutzen `r[p]`. Zufall nur über SplitMix64 (`vha_rng`), nie über Pythons `hash()`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from vha_rng import uniform_block

EPS = 1e-12


def seed_of(label: str) -> int:
    """64-Bit-Seed aus einer Bezeichnung (FNV-1a über die UTF-8-Bytes). Pythons `hash` ist je Prozess gesalzen und taugt nicht."""
    h = 0xCBF29CE484222325
    for byte in label.encode("utf-8"):
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


@lru_cache(maxsize=None)
def bundles(n: int, m: int) -> np.ndarray:
    """Alle `m ** n` Bündel als Zeilen (N, n); die Tabelle ist schreibgeschützt und wird je (n, m) nur einmal gebaut."""
    arr = np.indices((m,) * n).reshape(n, -1).T.copy()
    arr.setflags(write=False)
    return arr


class Domain:
    """Eine Verhandlungsdomäne. `w`: (2, n) Gewichte, `v`: Liste von zwei (n, m)-Wertlisten, `r`: Reservationsnutzen beider Seiten, `tie`: fester Zufallsvektor je Bündel (Gleichstandsregel)."""

    def __init__(self, w, v, r, tie):
        self.w = np.asarray(w, float)
        self.v = [np.asarray(v[0], float), np.asarray(v[1], float)]
        self.r = np.asarray(r, float)
        self.n, self.m = self.v[0].shape
        self.B = bundles(self.n, self.m)
        self.N = self.B.shape[0]
        ar = np.arange(self.n)[None, :]
        self.u = [(self.v[p][ar, self.B] * self.w[p]).sum(1) for p in (0, 1)]
        self.tie = np.asarray(tie, float)
        assert self.tie.shape == (self.N,)


def swap_sides(dom: Domain) -> Domain:
    """Dieselbe Domäne mit vertauschten Seiten (Nutzen, Gewichte, Wertlisten und Reservationsnutzen), gleiche Gleichstandsregel."""
    return Domain(dom.w[::-1], dom.v[::-1], dom.r[::-1], dom.tie)


def _norm_rows(x: np.ndarray) -> np.ndarray:
    lo = x.min(1, keepdims=True)
    hi = x.max(1, keepdims=True)
    span = hi - lo
    return np.where(span > 0, (x - lo) / np.where(span > 0, span, 1.0), 0.0)


def _dirichlet(u: np.ndarray) -> np.ndarray:
    """Zeilenweise Dirichlet(1)-Gewichte (Summe 1) aus Gleichverteilten."""
    e = -np.log(1.0 - u)
    return e / e.sum(axis=1, keepdims=True)


def make_domain(seed: int, n: int = 4, m: int = 4, s: float = 0.5, g: float = 0.5, r: float = 0.2) -> Domain:
    """Zufallsdomäne aus einem SplitMix64-Strom.

    `s` = Gewichtsunterschied: Gewichte der Seite p sind (1 − s)·c + s·x_p mit gemeinsamem c und unabhängigen x_p, alle Dirichlet(1) (0: gleiche Gewichte, 1: unabhängige).
    `g` = Gegensatz der Wertlisten: v₁ = g·(1 − v₀) + (1 − g)·Zufall, danach je Punkt auf 0 bis 1 gestreckt (0: unabhängig, 1: genau gegenläufig)."""
    count = 3 * n + 2 * n * m + m ** n
    u = uniform_block(seed, count)
    pos = 0

    def take(k):
        nonlocal pos
        out = u[pos:pos + k]
        pos += k
        return out

    c = _dirichlet(take(n).reshape(1, n))[0]
    a = _dirichlet(take(n).reshape(1, n))[0]
    b = _dirichlet(take(n).reshape(1, n))[0]
    w = np.vstack([(1 - s) * c + s * a, (1 - s) * c + s * b])
    v0 = _norm_rows(take(n * m).reshape(n, m))
    raw1 = take(n * m).reshape(n, m)
    v1 = _norm_rows(g * (1.0 - v0) + (1.0 - g) * raw1)
    tie = take(m ** n)
    return Domain(w, [v0, v1], [r, r], tie)


def weight_gap(dom: Domain) -> float:
    """Gewichtsabstand ½‖w₀ − w₁‖₁ (0: gleiche Gewichte)."""
    return float(0.5 * np.abs(dom.w[0] - dom.w[1]).sum())


# ---------------------------------------------------------------- Pareto, Nash
def pareto_idx(u0: np.ndarray, u1: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Nichtdominierte Bündel durch einen Sweep (nach u₀ absteigend, dann u₁ absteigend). Gleiche Nutzenpaare bleiben alle erhalten (auf 12 Stellen gerundet verglichen)."""
    ids = np.flatnonzero(mask) if mask is not None else np.arange(len(u0))
    a = np.round(u0[ids], 12)
    b = np.round(u1[ids], 12)
    order = np.lexsort((-b, -a))
    keep: list[int] = []
    best = -np.inf
    i = 0
    while i < len(order):
        j = i
        ai = a[order[i]]
        while j < len(order) and a[order[j]] == ai:
            j += 1
        grp = order[i:j]
        bmax = b[grp].max()
        if bmax > best:
            keep.extend(ids[grp[b[grp] == bmax]].tolist())
            best = bmax
        i = j
    return np.array(sorted(keep), dtype=int)


def dom_info(dom: Domain) -> dict | None:
    """Kennzahlen der Domäne: individuell rationale Bündel, deren Pareto-Grenze, Nash-Bündel (größtes Produkt der Gewinne über den Reservationsnutzen), größte Summe.
    `None`, wenn kein Bündel beiden Seiten mindestens den Reservationsnutzen bringt."""
    u0, u1 = dom.u
    ir = (u0 >= dom.r[0] - EPS) & (u1 >= dom.r[1] - EPS)
    if not ir.any():
        return None
    front = pareto_idx(u0, u1, ir)
    prod = np.where(ir, (u0 - dom.r[0]) * (u1 - dom.r[1]), -1.0)
    nash = int(np.argmax(prod))
    return dict(ir=ir, front=front, F=np.stack([u0[front], u1[front]], 1), nash=nash, nashprod=float(prod[nash]), maxsum=float((u0 + u1)[ir].max()), nir=int(ir.sum()))


def domain_set(label: str, count: int, **kw) -> tuple[list[tuple[Domain, dict]], int]:
    """`count` gültige Domänen (mit mindestens einem individuell rationalen Bündel) aus den Seeds `label|0`, `label|1`, …; zweites Ergebnis: Zahl verworfener Domänen."""
    out: list[tuple[Domain, dict]] = []
    skipped = 0
    inst = 0
    while len(out) < count:
        d = make_domain(seed_of(f"{label}|{inst}"), **kw)
        inst += 1
        info = dom_info(d)
        if info is None:
            skipped += 1
            continue
        out.append((d, info))
    return out, skipped
