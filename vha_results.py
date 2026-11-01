"""Zahlen aus den gespeicherten Messreihen (`data/messreihen.json`, erzeugt von `vha_experiments`): Mittel und Standardabweichung (n − 1) über die vier Zufalls-Seeds, Rangfolgen,
Varianzanteile. Keine Formatierung und keine Texte (das macht `vha_evaluation`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import vha_constants as C
import vha_experiments as XP
import vha_strategies as ST
import vha_tournament as TO

SEEDS = C.SEEDS
P6 = list(ST.POOL6)
P9 = list(ST.POOL9)
EFFECT_OPPS = XP.EFFECT_OPPONENTS
FACTORS = ("Kurve", "Akzeptanz", "Modell")
DECOMP_METRICS = {"own": "eigener Nutzen", "wel": "Summe der Nutzen", "dist": "Abstand zur Pareto-Grenze", "agree": "Einigungsrate"}


def _arr(x):
    """Liste mit `None` (NaN in der JSON-Datei) als Array mit NaN."""
    return np.array(x, dtype=float)


def _key(*job) -> str:
    return XP.job_key(tuple(job))


def _job(data: dict, *job):
    return data["jobs"][_key(*job)]


def _ms(values) -> tuple[float, float]:
    """(Mittel, Standardabweichung mit n − 1) einer Liste von Seed-Werten."""
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1))


# --------------------------------------------------------------------------------- Turnier
@dataclass
class CellStats:
    """Auswertung einer Zelle des Turniers: Mittel und Standardabweichung über die Seeds je Strategie und Kennzahl, Rang je Seed."""
    names: list
    mean: dict
    sd: dict
    own_by_seed: np.ndarray       # (Seeds, K)
    ranks: np.ndarray             # (Seeds, K): Rang 1 = größter eigener Nutzen
    top: int                      # Index der Strategie mit dem größten Mittel des eigenen Nutzens
    top_seeds: int                # in wie vielen Seeds sie auch einzeln auf Rang 1 liegt
    M: np.ndarray                 # (Seeds, K, K, 7): Paarmatrizen der Mitglieder


def cell_stats(Ms: np.ndarray, names: list, members: list[int]) -> CellStats:
    sc = [TO.strategy_scores(M, members) for M in Ms]
    mean = {k: np.mean([x[k] for x in sc], axis=0) for k in TO.SCORE_KEYS}
    sd = {k: np.std([x[k] for x in sc], axis=0, ddof=1) for k in TO.SCORE_KEYS}
    own = np.array([x["own"] for x in sc])
    ranks = np.array([TO.ranks_desc(o) for o in own])
    top = int(np.argmax(mean["own"]))
    return CellStats(names, mean, sd, own, ranks, top, int((ranks[:, top] == 1).sum()), Ms[:, members][:, :, members])


def turnier_matrices(data: dict, T: int, s: float) -> np.ndarray:
    return np.stack([_arr(_job(data, "turnier", T, float(s), sd)["M"]) for sd in SEEDS])


def turnier_cell(data: dict, T: int, s: float) -> CellStats:
    return cell_stats(turnier_matrices(data, T, s), P6, list(range(6)))


def rule_off_cell(data: dict) -> CellStats:
    """Dasselbe Turnier (T = 20, s = 0,5, dieselben Domänen) ohne die Regel „letzte Runde“."""
    return cell_stats(np.stack([_arr(_job(data, "regelaus", sd)["M"]) for sd in SEEDS]), P6, list(range(6)))


def endgame_cell(data: dict, T: int, pool: str = "9") -> CellStats:
    """Endspiel-Turnier (T = 20 oder 21, s = 0,5, 50 Domänen je Seed): Pool „6“ (Standard), „7“ (mit Endspiel-Ausnutzer) oder „9“ (mit den beiden Orakel-Agenten)."""
    Ms = np.stack([_arr(_job(data, "endspiel", T, sd)["M"]) for sd in SEEDS])
    members = list(range({"6": 6, "7": 7, "9": 9}[pool]))
    return cell_stats(Ms, [P9[i] for i in members], members)


def skipped_domains(data: dict) -> int:
    """Domänen ohne individuell rational mögliche Einigung, die beim Erzeugen der Turnier-Aufträge verworfen wurden (Summe über Zellen ohne Wiederholung für die Fristen)."""
    return int(sum(_job(data, "turnier", C.TURNIER_TS[0], float(s), sd)["skipped"] for s in C.TURNIER_SS for sd in SEEDS))


def winners(data: dict) -> dict:
    """(T, s) -> (Name des Siegers nach eigenem Nutzen, in wie vielen der vier Seeds er einzeln auf Rang 1 liegt)."""
    out = {}
    for T in C.TURNIER_TS:
        for s in C.TURNIER_SS:
            c = turnier_cell(data, T, s)
            out[(T, s)] = (P6[c.top], c.top_seeds)
    return out


def rank_correlation_with_reference(data: dict, T: int, s: float, ref=(20, 0.5)) -> float:
    return TO.rank_agreement(turnier_cell(data, T, s).mean["own"], turnier_cell(data, *ref).mean["own"])


def last_place_counts(data: dict) -> tuple[int, int, list]:
    """Conceder: in wie vielen der 9 Zellen im Mittel auf dem letzten Rang, in wie vielen der 36 (Zelle, Seed)-Paare, und die Zellen, in denen nicht."""
    ci = P6.index(ST.CONCEDER)
    last_mean = last_seed = 0
    exceptions = []
    for T in C.TURNIER_TS:
        for s in C.TURNIER_SS:
            c = turnier_cell(data, T, s)
            if int(np.argmin(c.mean["own"])) == ci:
                last_mean += 1
            else:
                exceptions.append((T, s))
            last_seed += int((c.ranks[:, ci] == len(P6)).sum())
    return last_mean, last_seed, exceptions


def selfplay_first_advantage(data: dict, T: int, s: float = 0.5) -> dict:
    """Nutzen der zuerst bietenden Position minus der zweiten im Selbstspiel je Strategie (Mittel über die Seeds)."""
    M = turnier_matrices(data, T, s).mean(0)
    return {n: float(M[k, k, TO.IX["u0"]] - M[k, k, TO.IX["u1"]]) for k, n in enumerate(P6)}


def pair_agreement(data: dict, T: int, a: str, b: str, s: float = 0.5) -> float:
    """Einigungsrate eines Paares (beide Positionen gemittelt; für a = b die Diagonale), Mittel über die Seeds."""
    M = turnier_matrices(data, T, s).mean(0)
    i, j = P6.index(a), P6.index(b)
    ix = TO.IX["agree"]
    return float(M[i, j, ix]) if i == j else float((M[i, j, ix] + M[j, i, ix]) / 2)


def pair_matrix(c: CellStats, metric: str) -> np.ndarray:
    """Mittel über die Seeds: Zeile = zuerst bietende Strategie, Spalte = zweite."""
    return c.M.mean(0)[:, :, TO.IX[metric]]


# --------------------------------------------------------------------------------- Position, Regel, Endspiel
def last_bidder(T: int) -> int:
    """Position, die in Runde T bietet: bei geradem T die zweite (1), bei ungeradem T die erste (0)."""
    return 1 if T % 2 == 0 else 0


def position_cell(data: dict, T: int, rule: bool, pos: int, opp: str) -> dict:
    """Hartnäckig in Position `pos` gegen `opp`: Nutzen der Hartnäckigen (Mittel, Std über die Seeds), Nutzen des Gegners, Einigungsrate, mittlere Einigungsrunde (nur Einigungen)."""
    cells = [_job(data, "position", T, int(rule), sd)["cells"][f"{pos}|{opp}"] for sd in SEEDS]
    mine = [c[0] for c in cells]
    rsum, rcnt = sum(c[3] for c in cells), sum(c[4] for c in cells)
    m, s = _ms(mine)
    return {"mine": m, "mine_sd": s, "theirs": float(np.mean([c[1] for c in cells])), "agree": float(np.mean([c[2] for c in cells])),
            "round": float(rsum / rcnt) if rcnt else float("nan"), "last": last_bidder(T) == pos}


def endgame_positions(data: dict, T: int, name: str = ST.ENDSPIEL) -> dict:
    """Nutzen des Programms `name` gegen Boulware als erste bzw. zweite Position (Mittel über die Seeds), dazu Boulware gegen Boulware in denselben Positionen."""
    Ms = np.stack([_arr(_job(data, "endspiel", T, sd)["M"]) for sd in SEEDS]).mean(0)
    k, b = P9.index(name), P9.index(ST.BOULWARE)
    return {"first": float(Ms[k, b, TO.IX["u0"]]), "second": float(Ms[b, k, TO.IX["u1"]]), "base_first": float(Ms[b, b, TO.IX["u0"]]), "base_second": float(Ms[b, b, TO.IX["u1"]])}


# --------------------------------------------------------------------------------- Gegnermodell
def model_curve(data: dict, opp: str, s: float) -> dict:
    """Güte des Häufigkeitsmodells nach k Angeboten des Gegners: Gewichtsabstand und Rangkorrelation (Mittel und Std über die Seeds je k), Abstand beim Raten gleicher Gewichte."""
    res = [_job(data, "lernmodell", opp, float(s), sd) for sd in SEEDS]
    gap = np.array([[r["gap"][str(k)] for k in C.MODEL_KS] for r in res])
    rho = np.array([[r["rho"][str(k)] for k in C.MODEL_KS] for r in res])
    return {"k": list(C.MODEL_KS), "gap": gap.mean(0), "gap_sd": gap.std(0, ddof=1), "rho": rho.mean(0), "rho_sd": rho.std(0, ddof=1), "guess": float(np.mean([r["guess"] for r in res]))}


def effect_cell(data: dict, curve: str, model: str) -> dict:
    """Wirkung des Gegnermodells (keins, haeufig, orakel) auf Boulware/Linear/Conceder gegen die fünf Basisgegner (T = 20, s = 0,5): Kennzahl -> (Mittel, Std über die Seeds)."""
    res = [_job(data, "wirkmodell", curve, model, sd) for sd in SEEDS]
    return {k: _ms([r[k] for r in res]) for k in ("own", "opp", "wel", "dist", "nr", "agree")}


def effect_gap_closed(data: dict, curve: str, key: str) -> float:
    """Anteil der Lücke zwischen „kein Modell“ und „Orakel“, den das Häufigkeitsmodell schließt (Mittel über die Seeds)."""
    k0, kh, ko = (effect_cell(data, curve, m)[key][0] for m in ("keins", "haeufig", "orakel"))
    return (kh - k0) / (ko - k0) if abs(ko - k0) > 1e-9 else float("nan")


# --------------------------------------------------------------------------------- Zerlegung
def decomp_combos() -> list[tuple]:
    """Die 48 Agenten in der Reihenfolge der Aufträge: Kurve × Akzeptanz × Modell."""
    return [(c, a, m) for c in XP.CURVES for a in XP.ACS for m in XP.MODELS]


def decomp_arrays(data: dict) -> dict:
    """Metrik -> Array (Seeds, 48 Agenten): Mittel des Agenten über 50 Domänen × 6 Gegner × 2 Positionen je Seed."""
    combos = decomp_combos()
    return {m: np.array([[_job(data, "zerlegung", c, a, mo, sd)[m] for (c, a, mo) in combos] for sd in SEEDS]) for m in DECOMP_METRICS}


def eta_squared(x: np.ndarray, combos: list[tuple], keep=None) -> dict:
    """Varianzanteil der Haupteffekte Kurve, Akzeptanz, Modell an den Agentenmitteln `x` (Seeds, Agenten): η² = SS_Faktor / SS_gesamt je Seed; Rückgabe Faktor -> (Mittel, Std über die Seeds),
    dazu „Rest“ (Wechselwirkungen). `keep(combo)` wählt die Stufen aus (Standard alle)."""
    idx = [i for i, cb in enumerate(combos) if keep is None or keep(cb)]
    sub = [combos[i] for i in idx]
    per = {f: [] for f in FACTORS}
    rest = []
    for si in range(x.shape[0]):
        v = x[si][idx]
        tot = ((v - v.mean()) ** 2).sum()
        used = 0.0
        for fi, f in enumerate(FACTORS):
            lab = np.array([cb[fi] for cb in sub])
            ss = sum((lab == lv).sum() * (v[lab == lv].mean() - v.mean()) ** 2 for lv in set(lab.tolist()))
            per[f].append(ss / tot)
            used += ss / tot
        rest.append(1.0 - used)
    out = {f: _ms(v) for f, v in per.items()}
    out["Rest"] = _ms(rest)
    return out


STAGE_SETS = {
    "alle 48 Agenten": None,
    "ohne „ab Reservation“": lambda cb: cb[1] != "ab Reservation",
    "ohne Tit-for-Tat": lambda cb: cb[0] != "Tit-for-Tat",
    "ohne beides (27 Agenten)": lambda cb: cb[1] != "ab Reservation" and cb[0] != "Tit-for-Tat",
}


def level_means(data: dict) -> dict:
    """(Faktor, Stufe) -> Mittel der Agenten mit dieser Stufe (alle Seeds, alle übrigen Stufen) je Metrik."""
    arrs, combos = decomp_arrays(data), decomp_combos()
    out = {}
    for fi, f in enumerate(FACTORS):
        for lv in dict.fromkeys(cb[fi] for cb in combos):
            sel = [i for i, cb in enumerate(combos) if cb[fi] == lv]
            out[(f, lv)] = {m: float(arrs[m].mean(0)[sel].mean()) for m in DECOMP_METRICS}
    return out


def accept_cells(data: dict) -> list[dict]:
    """Annahmeregeln im Einzelnen (Kurve Linear, kein Modell): Bezeichnung und Kennzahl -> (Mittel, Std über die Seeds)."""
    rows = []
    for k, (lab, *_rest) in enumerate(XP.ACCEPT_VARIANTS):
        res = [_job(data, "akzeptanz", k, sd) for sd in SEEDS]
        rows.append({"label": lab, **{key: _ms([r[key] for r in res]) for key in ("own", "opp", "wel", "agree")}})
    return rows


def best_worst_agents(data: dict, n: int = 5) -> tuple[list, list]:
    """Die n besten und n schlechtesten der 48 Agenten nach eigenem Nutzen (Mittel über die Seeds): (Kurve, Akzeptanz, Modell, Mittel, Std)."""
    arrs, combos = decomp_arrays(data), decomp_combos()
    mo, sd = arrs["own"].mean(0), arrs["own"].std(0, ddof=1)
    order = np.argsort(-mo)

    def pick(ids):
        return [(*combos[i], float(mo[i]), float(sd[i])) for i in ids]

    return pick(order[:n]), pick(order[-n:])


# --------------------------------------------------------------------------------- Robustheit
def robust_cells(data: dict) -> list[dict]:
    """Je geändertem Regler: Rangfolge nach eigenem Nutzen (Mittel über die Seeds), Sieger, Seeds mit demselben Sieger, Letzter, Einigung und Summe des Siegers."""
    rows = []
    for k, v in enumerate(XP.ROBUST):
        res = [_job(data, "robust", k, sd) for sd in SEEDS]
        names = res[0]["names"]
        own = np.array([r["own"] for r in res])
        m = own.mean(0)
        top, last = int(np.argmax(m)), int(np.argmin(m))
        rows.append({"label": v["label"], "names": names, "own": m, "top": names[top], "last": names[last], "wins": int(sum(int(np.argmax(o)) == top for o in own)),
                     "top_own": float(m[top]), "top_agree": float(np.mean([r["agree"][top] for r in res])), "top_wel": float(np.mean([r["wel"][top] for r in res]))})
    return rows


# --------------------------------------------------------------------------------- Lernender
def learner_summary(data: dict, episodes: int) -> dict:
    """Lernender nach `episodes` Episoden: Mittel über die sechs Gegner je Trainings-Seed (Mittel, Std, kleinster, größter), je Gegner (Mittel, Std über die Seeds), Einigungsrate."""
    res = [_job(data, "lerner", episodes, sd)["ref"] for sd in SEEDS]
    per_seed = np.array([np.mean([r[o][0] for o in P6]) for r in res])
    return {"mean": float(per_seed.mean()), "sd": float(per_seed.std(ddof=1)), "min": float(per_seed.min()), "max": float(per_seed.max()), "per_seed": per_seed,
            "per_opp": {o: _ms([r[o][0] for r in res]) for o in P6}, "agree": float(np.mean([r[o][1] for r in res for o in P6]))}


def learner_baselines(data: dict) -> dict:
    """Feste Strategien unter den Testbedingungen des Lernenden (Test-Domänen je Trainings-Seed): Name -> Mittel/Std über die Seeds und je Gegner."""
    res = [_job(data, "lerner_basis", sd)["ref"] for sd in SEEDS]
    out = {}
    for name in XP.LEARN_BASELINES:
        per_seed = np.array([np.mean(list(r[name].values())) for r in res])
        out[name] = {"mean": float(per_seed.mean()), "sd": float(per_seed.std(ddof=1)), "per_opp": {o: float(np.mean([r[name][o] for r in res])) for o in P6}}
    return out


def learner_transfer(data: dict) -> list[dict]:
    """Übertragung des mit 20.000 Episoden bei T = 20, s = 0,5 trainierten Lernenden auf andere Bedingungen (Mittel über die Trainings-Seeds und die sechs Gegner)."""
    lern = [_job(data, "lerner", C.LEARN_REF_EPISODES, sd)["transfer"] for sd in SEEDS]
    base = [_job(data, "lerner_basis", sd)["transfer"] for sd in SEEDS]
    rows = []
    for label, *_ in XP.TRANSFER:
        lm = np.array([x[label] for x in lern])
        bb = float(np.mean([b[label][ST.BOULWARE] for b in base]))
        bm = float(np.mean([b[label][ST.BOULWARE_MODELL] for b in base]))
        rows.append({"label": label, "lern": float(lm.mean()), "lern_sd": float(lm.std(ddof=1)), "boulware": bb, "modell": bm, "linear": float(np.mean([b[label][ST.LINEAR] for b in base])),
                     "ahead": int((lm > max(bb, bm)).sum())})
    return rows


def learner_omitted(data: dict) -> list[dict]:
    """Training ohne einen Gegner (20.000 Episoden): Nutzen gegen den ausgelassenen (Mittel, Std über die Seeds), gegen die vier übrigen, feste Strategien gegen den ausgelassenen."""
    rows = []
    for left in (ST.CONCEDER, ST.BOULWARE):
        res = [_job(data, "lerner_aus", left, sd) for sd in SEEDS]
        m, s = _ms([r["left"] for r in res])
        rows.append({"left": left, "mean": m, "sd": s, "others": float(np.mean([r["others"] for r in res])), "fixed": {n: float(np.mean([r["fixed"][n] for r in res])) for n in (ST.BOULWARE, ST.LINEAR, ST.CONCEDER)}})
    return rows
