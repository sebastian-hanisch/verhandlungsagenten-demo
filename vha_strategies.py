"""Bausteine der Verhandlungsprogramme: Zielnutzen (Gebotskurve), Wahl auf der Isolinie, Annahmeregel (die Zerlegung Gebot, Gegnermodell, Akzeptanz).

Jede Funktion ist eine Regel und hat nur ihre Eingaben. Alle Nutzen liegen auf [0, 1] (der größte Nutzen einer Seite ist 1); `r` ist der Reservationsnutzen der Seite.
Strategien sind Modellannahmen (Formelfamilie nach Faratin u. a. laut Sekundärliteratur, nicht im Volltext geprüft), keine Aussagen über echte Verhandler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12

CURVES = ("zeit", "hart", "tft")                       # Zeitkurve (Boulware, Linear, Conceder je nach β), Hartnäckig, Tit-for-Tat relativ
ACCEPTS = ("reserv", "schwelle", "zeit", "next")      # ab Reservation, feste Schwelle θ, linearer Zeitplan, nächstes eigenes Gebot
MODELS = ("keins", "haeufig", "orakel")               # Gegnermodell: keins, Häufigkeitsmodell, Orakel (kennt den wahren Nutzen des Gegners)


@dataclass(frozen=True)
class Spec:
    """Ein Verhandlungsprogramm als Baustein-Kombination. `beta`: Exponent der Zeitkurve (β < 1 Boulware, 1 linear, β > 1 Conceder); `theta`: Schwelle der Annahmeregel „Schwelle“;
    `gamma`: Anteil, mit dem Tit-for-Tat den Zuwachs des Gegners erwidert; `endgame`: in der letzten Runde das beste Bündel bieten, das der Gegner laut Modell noch annimmt."""
    curve: str = "zeit"
    beta: float = 0.2
    ac: str = "next"
    model: str = "keins"
    theta: float = 0.7
    gamma: float = 1.0
    endgame: bool = False


# Die Strategien des Pools (Anzeigename -> Bausteine). Reihenfolge = Reihenfolge in Tabellen und Auswahlfeldern.
HARTNAECKIG = "Hartnäckig"
BOULWARE = "Boulware"
LINEAR = "Linear"
CONCEDER = "Conceder"
TFT = "Tit-for-Tat (relativ)"
BOULWARE_MODELL = "Boulware + Modell"
ENDSPIEL = "Boulware + Modell + Endspiel"
POOL = {
    HARTNAECKIG: Spec(curve="hart"),
    BOULWARE: Spec(curve="zeit", beta=0.2),
    LINEAR: Spec(curve="zeit", beta=1.0),
    CONCEDER: Spec(curve="zeit", beta=5.0),
    TFT: Spec(curve="tft"),
    BOULWARE_MODELL: Spec(curve="zeit", beta=0.2, model="haeufig"),
    ENDSPIEL: Spec(curve="zeit", beta=0.2, model="haeufig", endgame=True),
}
POOL6 = (HARTNAECKIG, BOULWARE, LINEAR, CONCEDER, TFT, BOULWARE_MODELL)     # der Standard-Pool der Messreihen
POOL7 = POOL6 + (ENDSPIEL,)
# Obergrenzen für den Endspiel-Vergleich (kennen den wahren Nutzen des Gegners): nur in den Messreihen, nicht im Auswahlfeld der App
ORAKEL = "Boulware + Orakel"
ORAKEL_ENDSPIEL = "Boulware + Orakel + Endspiel"
EXTRA = {ORAKEL: Spec(curve="zeit", beta=0.2, model="orakel"), ORAKEL_ENDSPIEL: Spec(curve="zeit", beta=0.2, model="orakel", endgame=True)}
POOL9 = POOL7 + (ORAKEL, ORAKEL_ENDSPIEL)


def spec_of(name: str) -> Spec:
    return POOL[name] if name in POOL else EXTRA[name]


# ---------------------------------------------------------------- Gebot: Zielnutzen
def progress(t: int, horizon: int) -> float:
    """Zeitfortschritt τ = (t − 1) / (T − 1) auf [0, 1] (Runde 1: 0, letzte Runde: 1)."""
    return min(1.0, max(0.0, (t - 1.0) / (horizon - 1.0)))


def target_curve(t: int, horizon: int, r: float, beta: float) -> float:
    """Zielnutzen der Zeitkurve: 1 − (1 − r) · τ^(1/β). β < 1 hält lange hoch (Boulware), β = 1 sinkt gleichmäßig, β > 1 gibt früh nach (Conceder). Endet bei r."""
    return 1.0 - (1.0 - r) * progress(t, horizon) ** (1.0 / beta)


def tft_target(target: float, gain: float, gamma: float, r: float) -> float:
    """Tit-for-Tat (relativ): das Ziel sinkt um `gamma` mal den Zuwachs `gain`, den das neue Angebot des Gegners gegenüber seinem vorigen für mich brachte; nie unter r, nie bei Rückgang."""
    return max(r, target - gamma * gain) if gain > 0 else target


def window_choice(u: np.ndarray, target: float, band: float, est: np.ndarray | None, tie: np.ndarray) -> int:
    """Wahl auf der Isolinie: unter den Bündeln mit Nutzen in [Ziel, Ziel + Fenster] das mit dem größten **geschätzten Gegnernutzen** (ohne Modell: alle gleich), bei Gleichstand
    das mit dem kleinsten Zufallsvektor `tie`. Ist das Fenster leer: das Bündel mit dem kleinsten Nutzen über dem Ziel; liegt das Ziel über allen Nutzen: das beste Bündel."""
    mask = (u >= target - EPS) & (u <= target + band + EPS)
    if not mask.any():
        cand = u >= target - EPS
        if not cand.any():
            mask = u >= u.max() - EPS
        else:
            mask = cand & (u <= u[cand].min() + EPS)
    idxs = np.flatnonzero(mask)
    if est is not None:
        vals = est[idxs]
        idxs = idxs[vals >= vals.max() - EPS]
    return int(idxs[np.argmin(tie[idxs])])


def endgame_choice(u: np.ndarray, est: np.ndarray, r_opp: float, tie: np.ndarray) -> int | None:
    """Letztes Gebot des Endspiels: das für mich beste Bündel unter denen, die der Gegner laut Schätzung noch annimmt (est ≥ r_opp). `None`, wenn keins in Frage kommt.
    Bei einem Häufigkeitsmodell liegt `est` nicht auf der Nutzenskala des Gegners; der Vergleich mit r_opp ist dort eine grobe Annäherung."""
    ok = est >= r_opp - EPS
    if not ok.any():
        return None
    idxs = np.flatnonzero(ok)
    vals = u[idxs]
    idxs = idxs[vals >= vals.max() - EPS]
    return int(idxs[np.argmin(tie[idxs])])


# ---------------------------------------------------------------- Akzeptanz
def accept_rule(ac: str, u_offer: float, t: int, horizon: int, r: float, theta: float, u_next_own: float) -> bool:
    """Annahmeregel (ohne die Regel „letzte Runde“): „reserv“ ab Reservationsnutzen; „schwelle“ ab θ; „zeit“ ab dem linearen Zeitplan 1 − (1 − r)·τ;
    „next“, wenn das Angebot mindestens so viel bringt wie mein nächstes eigenes Gebot (`u_next_own`). Unter dem Reservationsnutzen wird nie angenommen."""
    if u_offer < r - EPS:
        return False
    if ac == "reserv":
        return True
    if ac == "schwelle":
        return u_offer >= theta - EPS
    if ac == "zeit":
        return u_offer >= 1.0 - (1.0 - r) * progress(t, horizon) - EPS
    if ac == "next":
        return u_offer >= u_next_own - EPS
    raise ValueError(ac)
