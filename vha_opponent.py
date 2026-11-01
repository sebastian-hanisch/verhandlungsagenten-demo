"""Häufigkeitsmodell des Gegners: aus den Angeboten des Gegners werden seine Wertlisten und Gewichte geschätzt.

Annahme des Modells (Modellannahme, keine Aussage über echte Verhandler): ein Gegner, der einen Wert eines Punktes oft anbietet, mag ihn; ein Punkt, an dem er von Angebot zu Angebot
den Wert wechselt, ist ihm weniger wichtig als einer, an dem er festhält.

* `counts[i][j]`: wie oft der Gegner Wert j von Punkt i angeboten hat; Wertschätzung `(counts + 1) / (max_j counts + 1)` (Glättung +1, der häufigste Wert bekommt 1);
* `nsame[i]`: wie oft Punkt i zwischen zwei aufeinanderfolgenden Angeboten gleich blieb; Gewicht `(1 + nsame[i]) / Σ (1 + nsame)`.
"""

from __future__ import annotations

import numpy as np


def new_state(n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Leeres Modell: Zähler (n, m) und Zahl gleich gebliebener Werte (n)."""
    return np.zeros((n, m)), np.zeros(n)


def update_counts(counts: np.ndarray, nsame: np.ndarray, offer: np.ndarray, prev: np.ndarray | None) -> None:
    """Ein beobachtetes Angebot (Bündel als Wertzeile) einarbeiten; `prev` ist das vorige Angebot desselben Gegners oder `None`. Verändert die Arrays an Ort und Stelle."""
    if prev is not None:
        nsame += offer == prev
    counts[np.arange(len(offer)), offer] += 1


def estimate_weights(nsame: np.ndarray) -> np.ndarray:
    """Geschätzte Gewichte des Gegners (Summe 1)."""
    h = 1.0 + nsame
    return h / h.sum()


def estimate_utility(counts: np.ndarray, nsame: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Geschätzter Nutzen des Gegners für jedes Bündel (Zeilen von `B`): Σ_i ŵ_i · v̂_i(b_i). Liegt auf einer eigenen Skala (kein Abgleich mit dem wahren Nutzen)."""
    vt = (counts + 1.0) / (counts.max(1, keepdims=True) + 1.0)
    return (vt[np.arange(B.shape[1])[None, :], B] * estimate_weights(nsame)).sum(1)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ränge mit mittleren Rängen bei Gleichstand (wie `scipy.stats.rankdata`, Rang 1 = kleinster Wert)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Rangkorrelation nach Spearman (Pearson-Korrelation der mittleren Ränge). Ist eine Seite konstant, ist sie nicht definiert: dann 0 (konstantes Modell ordnet nichts)."""
    ra, rb = _ranks(np.asarray(a, float)), _ranks(np.asarray(b, float))
    da, db = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((da * da).sum() * (db * db).sum()))
    return float((da * db).sum() / den) if den > 0 else 0.0


def weight_error(w_hat: np.ndarray, w_true: np.ndarray) -> float:
    """Gewichtsabstand ½‖ŵ − w‖₁ (0: genau, 1: völlig falsch)."""
    return float(0.5 * np.abs(np.asarray(w_hat) - np.asarray(w_true)).sum())
