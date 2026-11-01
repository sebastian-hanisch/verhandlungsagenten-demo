"""Messreihen: Turnier-Raster, Endspiel, Position, Gegnermodell, Zerlegung, Robustheit und Lernender, vorgerechnet in `data/messreihen.json`.

Eine **Zelle** besteht aus `C.SEEDS` (vier) unabhängigen Zufalls-Seeds; jeder Seed hat seine eigenen Domänen (`vha_domain.domain_set` aus der Bezeichnung des Auftrags, FNV-1a, SplitMix64).
Jeder **Auftrag** (`job`, ein Tupel) ist eine kleine Rechnung, die auf jedem Rechner dieselben Zahlen liefert; `tools/make_grid.py` verteilt sie auf mehrere Prozesse. Die Datei trägt den Hash der
Quellen, aus denen sie stammt (`generator_hash`); stimmt er nicht, zeigt die App die Reihen nicht (`load`). Ausgewertet wird in `vha_evaluation.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import vha_constants as C
import vha_domain as D
import vha_learning as LR
import vha_negotiation as NG
import vha_opponent as OPP
import vha_strategies as ST
import vha_tournament as TO

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "messreihen.json"
HASHED_FILES = ("vha_domain.py", "vha_strategies.py", "vha_opponent.py", "vha_negotiation.py", "vha_tournament.py", "vha_learning.py", "vha_experiments.py", "vha_rng.py", "vha_constants.py")
HASH_END = "# --- ab hier nicht mehr im Erzeuger-Hash"      # in vha_constants.py: nur der Teil davor (Messparameter) gehört zum Hash
SCHEMA = 1


def generator_hash() -> str:
    """Hash der Quelldateien, von denen die gespeicherten Messreihen abhängen (Zeilenenden vereinheitlicht)."""
    h = hashlib.sha256()
    for name in HASHED_FILES:
        text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        if name == "vha_constants.py":
            text = text.split(HASH_END)[0]
        h.update(name.encode() + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- Bezeichnungen und Bausteine der Messreihen
def sname(s: float) -> str:
    return f"{s:g}"


CURVES = {"Boulware": ST.Spec(curve="zeit", beta=0.2), "Linear": ST.Spec(curve="zeit", beta=1.0), "Conceder": ST.Spec(curve="zeit", beta=5.0), "Tit-for-Tat": ST.Spec(curve="tft")}
ACS = {"ab Reservation": "reserv", "Schwelle 0,7": "schwelle", "Zeitplan": "zeit", "nächstes Gebot": "next"}
MODELS = {"keins": "keins", "Häufigkeit": "haeufig", "Orakel": "orakel"}
EFFECT_CURVES = {"Boulware": 0.2, "Linear": 1.0, "Conceder": 5.0}
EFFECT_OPPONENTS = (ST.HARTNAECKIG, ST.BOULWARE, ST.LINEAR, ST.CONCEDER, ST.TFT)
POSITION_OPPONENTS = (ST.BOULWARE, ST.LINEAR, ST.CONCEDER, ST.TFT, ST.HARTNAECKIG)
# Akzeptanz im Einzelnen (Kurve Linear, kein Modell): (Bezeichnung, Annahmeregel, Schwelle, Regel „letzte Runde“)
ACCEPT_VARIANTS = (
    ("ab Reservation", "reserv", 0.7, True), ("Schwelle 0,5", "schwelle", 0.5, True), ("Schwelle 0,7", "schwelle", 0.7, True), ("Schwelle 0,9", "schwelle", 0.9, True),
    ("Zeitplan linear", "zeit", 0.7, True), ("nächstes Gebot", "next", 0.7, True),
    ("Schwelle 0,9, ohne Regel „letzte Runde“", "schwelle", 0.9, False), ("nächstes Gebot, ohne Regel „letzte Runde“", "next", 0.7, False),
    ("ab Reservation, ohne Regel „letzte Runde“", "reserv", 0.7, False),
)


def _pool_specs(changes: dict | None = None) -> dict[str, ST.Spec]:
    """Spezifikationen des Standard-Pools mit geänderten Bausteinen (`{Name: {Feld: Wert}}`)."""
    out = {}
    for name in ST.POOL6:
        spec = ST.POOL[name]
        if changes and name in changes:
            spec = ST.Spec(**{**spec.__dict__, **changes[name]})
        out[name] = spec
    return out


# Robustheit: ein Regler wird geändert, alles andere bleibt wie in der Basis (T = 20, s = 0,5, n = m = 4, r = 0,2, g = 0,5, Fenster 0,05).
ROBUST = (
    {"label": "Basis"},
    {"label": "Boulware-Exponent 0,1", "changes": {"Boulware": {"beta": 0.1}, "Boulware + Modell": {"beta": 0.1}}},
    {"label": "Boulware-Exponent 0,5", "changes": {"Boulware": {"beta": 0.5}, "Boulware + Modell": {"beta": 0.5}}},
    {"label": "Conceder-Exponent 2", "changes": {"Conceder": {"beta": 2.0}}},
    {"label": "Conceder-Exponent 10", "changes": {"Conceder": {"beta": 10.0}}},
    {"label": "Frist T = 21", "T": 21},
    {"label": "Frist T = 7", "T": 7},
    {"label": "Frist T = 6", "T": 6},
    {"label": "Fenster 0,02", "band": 0.02},
    {"label": "Fenster 0,10", "band": 0.10},
    {"label": "3 Punkte × 4 Werte", "dom": {"n": 3, "m": 4}},
    {"label": "5 Punkte × 3 Werte", "dom": {"n": 5, "m": 3}},
    {"label": "4 Punkte × 3 Werte", "dom": {"n": 4, "m": 3}},
    {"label": "2 Punkte × 5 Werte", "dom": {"n": 2, "m": 5}},
    {"label": "Reservationsnutzen 0", "dom": {"r": 0.0}},
    {"label": "Reservationsnutzen 0,5", "dom": {"r": 0.5}},
    {"label": "Gegensatz g = 0", "dom": {"g": 0.0}},
    {"label": "Gegensatz g = 1", "dom": {"g": 1.0}},
)
TRANSFER = (("T = 10", 10, 0.5, True), ("T = 21", 21, 0.5, True), ("T = 40", 40, 0.5, True), ("Regel „letzte Runde“ aus", 20, 0.5, False), ("s = 0", 20, 0.0, True), ("s = 1", 20, 1.0, True))
LEARN_BASELINES = (ST.BOULWARE, ST.BOULWARE_MODELL, ST.LINEAR, ST.CONCEDER)


# ---------------------------------------------------------------- Aufträge
def all_jobs() -> list[tuple]:
    jobs: list[tuple] = []
    for seed in C.SEEDS:
        for s in C.TURNIER_SS:
            for T in C.TURNIER_TS:
                jobs.append(("turnier", T, s, seed))
        jobs.append(("regelaus", seed))
        for T in C.ENDSPIEL_TS:
            jobs.append(("endspiel", T, seed))
        for T in C.POSITION_TS:
            for rule in (1, 0):
                jobs.append(("position", T, rule, seed))
        for s in C.MODEL_SS:
            for opp in EFFECT_OPPONENTS:
                jobs.append(("lernmodell", opp, s, seed))
        for curve in EFFECT_CURVES:
            for model in MODELS.values():
                jobs.append(("wirkmodell", curve, model, seed))
        for curve in CURVES:
            for ac in ACS:
                for model in MODELS:
                    jobs.append(("zerlegung", curve, ac, model, seed))
        for k in range(len(ACCEPT_VARIANTS)):
            jobs.append(("akzeptanz", k, seed))
        for k in range(len(ROBUST)):
            jobs.append(("robust", k, seed))
        for E in C.LEARN_EPISODES:
            jobs.append(("lerner", E, seed))
        jobs.append(("lerner_basis", seed))
        for left in (ST.CONCEDER, ST.BOULWARE):
            jobs.append(("lerner_aus", left, seed))
    # die schwersten zuerst: besser verteilt
    weight = {"lerner": 3, "lerner_aus": 2, "turnier": 1, "endspiel": 1}
    jobs.sort(key=lambda j: -weight.get(j[0], 0) * (j[1] if j[0] == "lerner" else 1))
    return jobs


def job_key(job: tuple) -> str:
    return "|".join(f"{x:g}" if isinstance(x, float) else str(x) for x in job)


def _num(x):
    """JSON-fähig: numpy-Zahlen in Python-Zahlen, NaN als None, sechs Nachkommastellen."""
    if isinstance(x, dict):
        return {str(k): _num(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    if isinstance(x, np.ndarray):
        return _num(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    return x


def _means(rows: list[dict], keys) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def _score_rows(me_spec, opponents, doms, horizon, rule=True, band=NG.BAND, keys=("own", "opp", "wel", "dist", "nr", "agree")) -> dict:
    """Eigener Nutzen usw. eines Programms gegen die Gegner, beide Positionen, gemittelt über die Domänen."""
    rows = []
    for d, info in doms:
        for opp in opponents:
            for pos in (0, 1):
                me, op = NG.make_agent(me_spec), NG.make_agent(opp)
                res = NG.negotiate(d, me, op, horizon, rule, band) if pos == 0 else NG.negotiate(d, op, me, horizon, rule, band)
                e = NG.evaluate(d, info, res)
                rows.append(dict(own=e["u0"] if pos == 0 else e["u1"], opp=e["u1"] if pos == 0 else e["u0"], wel=e["wel"], dist=e["dist"], nr=e["nashrel"], agree=1.0 if e["agree"] else 0.0))
    return _means(rows, keys)


def run_job(job: tuple):
    """Rechnet einen Auftrag; Rückgabe `(Schlüssel, Ergebnis)` (Ergebnis JSON-fähig)."""
    kind = job[0]
    out = _RUNNERS[kind](*job[1:])
    return job_key(job), _num(out)


def _turnier(T, s, seed):
    doms, skipped = D.domain_set(f"turnier|{sname(s)}|{seed}", C.TURNIER_DOMAINS, s=s)
    return {"M": TO.pair_means(ST.POOL6, doms, T), "skipped": skipped}


def _regelaus(seed):
    doms, skipped = D.domain_set(f"turnier|0.5|{seed}", C.TURNIER_DOMAINS, s=0.5)
    return {"M": TO.pair_means(ST.POOL6, doms, 20, last_round_rule=False), "skipped": skipped}


def _endspiel(T, seed):
    doms, skipped = D.domain_set(f"endspiel|{seed}", C.ENDSPIEL_DOMAINS, s=0.5)
    return {"M": TO.pair_means(ST.POOL9, doms, T), "skipped": skipped}


def _position(T, rule, seed):
    """Hartnäckig gegen die fünf anderen, getrennt nach Position; je Zelle (Nutzen der Hartnäckigen, Nutzen des Gegners, Einigungsrate, Summe der Einigungsrunden, Zahl der Einigungen)."""
    doms, _ = D.domain_set(f"position|{seed}", C.POSITION_DOMAINS, s=0.5)
    cells = {}
    for pos in (0, 1):
        for opp in POSITION_OPPONENTS:
            mine, theirs, ag, rsum, rcnt = [], [], [], 0.0, 0
            for d, _ in doms:
                h, o = NG.make_agent(ST.HARTNAECKIG), NG.make_agent(opp)
                res = NG.negotiate(d, h, o, T, bool(rule)) if pos == 0 else NG.negotiate(d, o, h, T, bool(rule))
                mine.append(res.u0 if pos == 0 else res.u1)
                theirs.append(res.u1 if pos == 0 else res.u0)
                ag.append(1.0 if res.agree else 0.0)
                if res.agree:
                    rsum += res.t
                    rcnt += 1
            cells[f"{pos}|{opp}"] = [float(np.mean(mine)), float(np.mean(theirs)), float(np.mean(ag)), rsum, rcnt]
    return {"cells": cells}


def model_quality(dom: D.Domain, offers, k: int, pos_obs: int = 1, horizon: int = C.MODEL_HORIZON) -> tuple[float, float]:
    """Güte des Häufigkeitsmodells eines Beobachters nach den ersten k Angeboten des Gegners: (Gewichtsabstand, Rangkorrelation von û und wahrem Gegnernutzen über alle Bündel)."""
    obs = NG.Agent(ST.Spec(curve="zeit", beta=1.0, ac="never", model="haeufig"))
    obs.reset(dom, pos_obs, horizon, False)
    for o in offers[:k]:
        obs.observe(o)
    return OPP.weight_error(obs.est_weights(), dom.w[1 - pos_obs]), OPP.rank_correlation(obs.est(), dom.u[1 - pos_obs])


def _lernmodell(opp, s, seed):
    doms, _ = D.domain_set(f"lernmodell|{sname(s)}|{seed}", C.MODEL_DOMAINS, s=s)
    gaps = {k: [] for k in C.MODEL_KS}
    rhos = {k: [] for k in C.MODEL_KS}
    guess = []
    for d, _ in doms:
        spec = ST.POOL[opp]
        a0 = NG.Agent(ST.Spec(**{**spec.__dict__, "ac": "never"}))
        a1 = NG.Agent(ST.Spec(curve="zeit", beta=1.0, ac="never", model="keins"))
        NG.negotiate(d, a0, a1, C.MODEL_HORIZON, False)
        offers = list(a1.opp)
        for k in C.MODEL_KS:
            g_, r_ = model_quality(d, offers, k)
            gaps[k].append(g_)
            rhos[k].append(r_)
        guess.append(OPP.weight_error(np.ones(d.n) / d.n, d.w[0]))
    return {"gap": {k: float(np.mean(v)) for k, v in gaps.items()}, "rho": {k: float(np.mean(v)) for k, v in rhos.items()}, "guess": float(np.mean(guess))}


def _wirkmodell(curve, model, seed):
    doms, _ = D.domain_set(f"wirkmodell|{seed}", C.EFFECT_DOMAINS, s=0.5)
    return _score_rows(ST.Spec(curve="zeit", beta=EFFECT_CURVES[curve], ac="next", model=model), EFFECT_OPPONENTS, doms, C.EFFECT_HORIZON)


def _zerlegung(curve, ac, model, seed):
    doms, _ = D.domain_set(f"zerlegung|{seed}", C.DECOMP_DOMAINS, s=0.5)
    spec = ST.Spec(**{**CURVES[curve].__dict__, "ac": ACS[ac], "theta": C.THETA, "model": MODELS[model]})
    return _score_rows(spec, ST.POOL6, doms, C.DECOMP_HORIZON, keys=("own", "wel", "dist", "agree"))


def _akzeptanz(k, seed):
    label, ac, theta, rule = ACCEPT_VARIANTS[k]
    doms, _ = D.domain_set(f"zerlegung|{seed}", C.DECOMP_DOMAINS, s=0.5)
    spec = ST.Spec(curve="zeit", beta=1.0, ac=ac, theta=theta, model="keins")
    return _score_rows(spec, ST.POOL6, doms, C.DECOMP_HORIZON, rule=rule, keys=("own", "opp", "wel", "agree"))


def _robust(k, seed):
    v = ROBUST[k]
    specs = _pool_specs(v.get("changes"))
    doms, _ = D.domain_set(f"robust|{seed}", C.ROBUST_DOMAINS, **{"s": 0.5, **v.get("dom", {})})
    M = TO.pair_means(list(specs.values()), doms, v.get("T", C.ROBUST_HORIZON), True, v.get("band", NG.BAND))
    sc = TO.strategy_scores(M)
    return {"names": list(specs), "own": sc["own"], "wel": sc["wel"], "agree": sc["agree"]}


def _lerner(E, seed):
    Q = LR.train(seed, E, LR.BASE5, C.LEARN_HORIZON)
    doms, _ = D.domain_set(f"lern|test|{seed}|0.5", C.LEARN_TEST_DOMAINS, s=0.5)
    ref = LR.test_against(Q, ST.POOL6, doms, C.LEARN_HORIZON)
    out = {"ref": {k: list(v) for k, v in ref.items()}}
    if E == C.LEARN_REF_EPISODES:
        trans = {}
        for label, T, s, rule in TRANSFER:
            d2, _ = D.domain_set(f"lern|test|{seed}|{sname(s)}", C.LEARN_TEST_DOMAINS, s=s)
            r2 = LR.test_against(Q, ST.POOL6, d2, T, rule)
            trans[label] = float(np.mean([v[0] for v in r2.values()]))
        out["transfer"] = trans
    return out


def _lerner_basis(seed):
    """Feste Strategien unter den Bedingungen des Lernenden: Referenz (je Gegner) und die Übertragungsbedingungen (Mittel über die sechs Gegner)."""
    doms, _ = D.domain_set(f"lern|test|{seed}|0.5", C.LEARN_TEST_DOMAINS, s=0.5)
    out = {"ref": {name: LR.baseline(name, ST.POOL6, doms, C.LEARN_HORIZON) for name in LEARN_BASELINES}, "transfer": {}}
    for label, T, s, rule in TRANSFER:
        d2, _ = D.domain_set(f"lern|test|{seed}|{sname(s)}", C.LEARN_TEST_DOMAINS, s=s)
        out["transfer"][label] = {name: float(np.mean(list(LR.baseline(name, ST.POOL6, d2, T, rule).values()))) for name in LEARN_BASELINES}
    return out


def _lerner_aus(left, seed):
    """Training ohne einen Gegner (20.000 Episoden), Test gegen den ausgelassenen und gegen die übrigen vier; dazu feste Strategien gegen den ausgelassenen."""
    pool = tuple(p for p in LR.BASE5 if p != left)
    Q = LR.train(seed, C.LEARN_REF_EPISODES, pool, C.LEARN_HORIZON)
    doms, _ = D.domain_set(f"lern|test|{seed}|0.5", C.LEARN_TEST_DOMAINS, s=0.5)
    t = LR.test_against(Q, (left,) + pool, doms, C.LEARN_HORIZON)
    fixed = {name: LR.baseline(name, (left,), doms, C.LEARN_HORIZON)[left] for name in (ST.BOULWARE, ST.LINEAR, ST.CONCEDER)}
    return {"left": t[left][0], "others": float(np.mean([t[p][0] for p in pool])), "fixed": fixed}


_RUNNERS = {"turnier": _turnier, "regelaus": _regelaus, "endspiel": _endspiel, "position": _position, "lernmodell": _lernmodell, "wirkmodell": _wirkmodell,
            "zerlegung": _zerlegung, "akzeptanz": _akzeptanz, "robust": _robust, "lerner": _lerner, "lerner_basis": _lerner_basis, "lerner_aus": _lerner_aus}


# ---------------------------------------------------------------- Speichern und Laden
def assemble(results: dict) -> dict:
    return {"schema": SCHEMA, "generator_hash": generator_hash(), "seeds": list(C.SEEDS), "jobs": results}


def write(payload: dict) -> None:
    DATA_FILE.parent.mkdir(exist_ok=True)
    DATA_FILE.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True), encoding="utf-8")


def load():
    """Die gespeicherten Messreihen oder `None` (Datei fehlt oder gehört zu einem anderen Rechencode)."""
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("schema") != SCHEMA or data.get("generator_hash") != generator_hash():
        return None
    return data
