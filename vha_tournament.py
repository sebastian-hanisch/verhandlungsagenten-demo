"""Turnier: jede Strategie gegen jede in beiden Positionen über viele Domänen; Kennzahlen je Strategie und je Paar.

`pair_means` liefert eine Matrix (K, K, 7): Zeile i bietet als Position 0 (zuerst), Spalte j als Position 1. Die Kennzahlen (letzte Achse, `METRICS`):
Nutzen von Position 0 und 1, Summe, Abstand zur Pareto-Grenze, relatives Nash-Produkt, Einigungsrate, mittlere Einigungsrunde (nur über Einigungen; `nan`, wenn keine).
Der **Wert einer Strategie** ist das Mittel über alle anderen Strategien des Pools und beide Positionen (das Selbstspiel steht getrennt in der Diagonale).
"""

from __future__ import annotations

import numpy as np

import vha_negotiation as NG

METRICS = ("u0", "u1", "wel", "dist", "nashrel", "agree", "t")
IX = {m: k for k, m in enumerate(METRICS)}
SCORE_KEYS = ("own", "opp", "wel", "dist", "nashrel", "agree")


def pair_means(names, doms, horizon: int, last_round_rule: bool = True, band: float = NG.BAND) -> np.ndarray:
    """Mittel über die Domänen für jedes geordnete Paar (zuerst bietende Strategie, zweite Strategie). `names`: Namen aus `vha_strategies.POOL`/`EXTRA` oder `Spec`-Objekte."""
    K = len(names)
    out = np.zeros((K, K, len(METRICS)))
    for i in range(K):
        for j in range(K):
            acc = np.zeros(6)
            tsum, tcnt = 0.0, 0
            for d, info in doms:
                res = NG.negotiate(d, NG.make_agent(names[i]), NG.make_agent(names[j]), horizon, last_round_rule, band)
                e = NG.evaluate(d, info, res)
                acc += (e["u0"], e["u1"], e["wel"], e["dist"], e["nashrel"], 1.0 if e["agree"] else 0.0)
                if e["agree"]:
                    tsum += e["t"]
                    tcnt += 1
            out[i, j, :6] = acc / len(doms)
            out[i, j, 6] = tsum / tcnt if tcnt else np.nan
    return out


def strategy_scores(M: np.ndarray, members: list[int] | None = None) -> dict[str, np.ndarray]:
    """Je Strategie X (aus `members`, Standard alle): Mittel über alle anderen Mitglieder Y und beide Positionen. Rückgabe: Kennzahl -> Vektor über die Mitglieder."""
    members = list(range(M.shape[0])) if members is None else list(members)
    res = {k: np.zeros(len(members)) for k in SCORE_KEYS}
    for a, x in enumerate(members):
        acc = {k: [] for k in SCORE_KEYS}
        for y in members:
            if y == x:
                continue
            acc["own"].append((M[x, y, IX["u0"]] + M[y, x, IX["u1"]]) / 2)
            acc["opp"].append((M[x, y, IX["u1"]] + M[y, x, IX["u0"]]) / 2)
            for key, mk in (("wel", "wel"), ("dist", "dist"), ("nashrel", "nashrel"), ("agree", "agree")):
                acc[key].append((M[x, y, IX[mk]] + M[y, x, IX[mk]]) / 2)
        for k in SCORE_KEYS:
            res[k][a] = float(np.mean(acc[k]))
    return res


def ranks_desc(v) -> np.ndarray:
    """Rang 1 = größter Wert (bei Gleichstand nach Reihenfolge)."""
    order = np.argsort(-np.asarray(v, float), kind="stable")
    r = np.empty(len(v), dtype=int)
    r[order] = np.arange(1, len(v) + 1)
    return r


def rank_agreement(a, b) -> float:
    """Rangkorrelation zweier Rangfolgen (Spearman über die Ränge der Werte, ohne Gleichstandskorrektur: Mittelwerte über viele Domänen sind praktisch nie gleich)."""
    ra, rb = ranks_desc(a).astype(float), ranks_desc(b).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])
