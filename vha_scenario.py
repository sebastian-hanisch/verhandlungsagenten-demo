"""Die Einstellungen der App als unveränderliche Datenklasse (Zwischenspeicher-Schlüssel) und die daraus gebaute Domäne."""

from __future__ import annotations

from dataclasses import dataclass

import vha_constants as C
import vha_domain as D


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    s: float = C.DEFAULT_S
    g: float = C.DEFAULT_G
    r: float = C.DEFAULT_R
    T: int = C.DEFAULT_T
    band: float = C.DEFAULT_BAND
    rule: str = C.DEFAULT_RULE
    first: str = C.BASE["first"]
    second: str = C.BASE["second"]
    seed: int = C.DEFAULT_SEED

    @property
    def rule_on(self) -> bool:
        return self.rule == "an"

    @property
    def bundles(self) -> int:
        return self.m ** self.n


def point_names(n: int) -> list[str]:
    """Namen der Punkte: die ersten n der festen Liste (Preis, Lieferfenster, Zahlungsziel, Haftung, Rückladung)."""
    return list(C.POINT_NAMES[:n])


def build_domain(st: Settings) -> D.Domain:
    """Die Domäne der Einstellungen (Zufalls-Seed der Domäne, Punkte, Werte, Gewichtsunterschied, Gegensatz, Reservationsnutzen)."""
    return D.make_domain(D.seed_of(f"app|{st.seed}"), n=st.n, m=st.m, s=st.s, g=st.g, r=st.r)
