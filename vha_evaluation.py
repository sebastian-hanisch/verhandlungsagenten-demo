"""Auswertung: Kennzahlen der Einstellungen (live), Zeilenbauer der Tabellen und die Verdikte.

Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`. Zahlen in Texten kommen aus der Rechnung (f-Strings), nicht aus dem Kopf.
Die Zahlen aus den gespeicherten Messreihen (Mittel, Standardabweichung über vier Zufalls-Seeds, Rangfolgen) liefert `vha_results`; hier werden sie formatiert und in Sätze gefasst.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import vha_constants as C
import vha_domain as D
import vha_negotiation as NG
import vha_opponent as OPP
import vha_results as R
import vha_strategies as ST
import vha_tournament as TO
from vha_scenario import Settings, build_domain

P6 = R.P6
P9 = R.P9
SHORT = {ST.TFT: "Tit-for-Tat", ST.BOULWARE_MODELL: "Boulware + Modell", ST.ENDSPIEL: "Endspiel-Ausnutzer", ST.ORAKEL: "Boulware + Orakel", ST.ORAKEL_ENDSPIEL: "Boulware + Orakel + Endspiel"}
TOP = ST.BOULWARE_MODELL
SPITZE = (ST.BOULWARE_MODELL, ST.BOULWARE, ST.TFT)       # die Spitzengruppe (Boulware + Modell, Boulware, Tit-for-Tat)


def short(name: str) -> str:
    return SHORT.get(name, name)


# --------------------------------------------------------------------------------- Formate
def de(x: float, nd: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und echtem Minuszeichen (−)."""
    text = f"{x:.{nd}f}"
    if float(text) == 0.0:
        text = text.lstrip("-")
    return text.replace(".", ",").replace("-", "−")


def pct(x: float, nd: int = 1) -> str:
    """Anteil als Prozentzahl mit Komma und Leerzeichen vor dem Prozentzeichen."""
    return f"{de(100.0 * x, nd)} %"


def signed(x: float, nd: int = 3) -> str:
    text = de(x, nd)
    return text if text.startswith("−") or float(text.replace(",", ".")) == 0.0 else "+" + text


def pm(mean: float, sd: float, nd: int = 3) -> str:
    """Mittel mit Standardabweichung über die vier Zufalls-Seeds in Klammern."""
    return f"{de(mean, nd)} ({de(sd, nd)})"


def thousands(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def stage_label(T: int, s: float) -> str:
    return f"T = {T}, s = {de(s, 1)}"


# --------------------------------------------------------------------------------- Live: eine Aushandlung
@dataclass
class LiveView:
    dom: D.Domain
    info: dict | None
    st: Settings
    result: NG.Result
    other: NG.Result            # dieselbe Aushandlung mit der anderen Einstellung der Regel „letzte Runde“
    trace: list
    ev: dict | None


def analyse(st: Settings) -> LiveView:
    """Die Domäne der Einstellungen, die Aushandlung der beiden gewählten Strategien (der Verlader bietet zuerst) und dieselbe mit der anderen Einstellung der Regel „letzte Runde“."""
    dom = build_domain(st)
    info = D.dom_info(dom)
    trace: list = []
    res = NG.negotiate(dom, NG.make_agent(st.first), NG.make_agent(st.second), st.T, st.rule_on, st.band, trace)
    other = NG.negotiate(dom, NG.make_agent(st.first), NG.make_agent(st.second), st.T, not st.rule_on, st.band)
    return LiveView(dom, info, st, res, other, trace, NG.evaluate(dom, info, res) if info is not None else None)


def bundle_label(dom: D.Domain, b: int) -> str:
    """Das Bündel als Wertstufen je Punkt, z. B. „2-0-3-1“ (Stufe = Index in der Wertliste, 0 bis m − 1)."""
    return "-".join(str(int(x)) for x in dom.B[b])


def trace_rows(view: LiveView) -> list[dict]:
    """Zeilen der Gebotsfolge: Runde, Bietender, Bündel, Nutzen beider Seiten, Ziel des Bietenden, Antwort."""
    rows = []
    for t, p, o, acc, goal in view.trace:
        rows.append({"Runde": str(t), "Bietet": "Verlader" if p == 0 else "Spediteur", "Bündel": bundle_label(view.dom, o),
                     "Nutzen Verlader": de(float(view.dom.u[0][o]), 3), "Nutzen Spediteur": de(float(view.dom.u[1][o]), 3),
                     "Ziel des Bietenden": "–" if goal != goal else de(goal, 3), "Antwort": "angenommen" if acc else "abgelehnt"})
    return rows


def summary_rows(view: LiveView) -> list[dict]:
    """Kennzahlen der Aushandlung mit Bezug zur Domäne (Nash, größte Summe, Pareto-Grenze der individuell rationalen Bündel)."""
    if view.info is None or view.ev is None:
        return []
    r, e, info, dom = view.result, view.ev, view.info, view.dom
    nb = info["nash"]
    return [
        {"Kennzahl": "Ausgang", "Wert": f"Einigung in Runde {r.t} von {view.st.T}" if r.agree else f"keine Einigung in {view.st.T} Runden (beide bekommen den Reservationsnutzen {de(float(dom.r[0]), 2)})"},
        {"Kennzahl": "Nutzen Verlader / Spediteur", "Wert": f"{de(r.u0, 3)} / {de(r.u1, 3)}"},
        {"Kennzahl": "Summe der Nutzen", "Wert": f"{de(e['wel'], 3)} (größte Summe aller rationalen Bündel: {de(info['maxsum'], 3)})"},
        {"Kennzahl": "Abstand zur Pareto-Grenze", "Wert": de(e["dist"], 3)},
        {"Kennzahl": "Nash-Lösung (Verlader / Spediteur)", "Wert": f"{de(float(dom.u[0][nb]), 3)} / {de(float(dom.u[1][nb]), 3)}"},
        {"Kennzahl": "Nash-Produkt relativ zur Nash-Lösung", "Wert": de(e["nashrel"], 3)},
        {"Kennzahl": "Bündel / individuell rational / auf der Grenze", "Wert": f"{thousands(dom.N)} / {thousands(info['nir'])} / {len(info['front'])}"},
    ]


def verdict_negotiation(view: LiveView) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: Einigung oder nicht, wie weit vom besten Punkt, und was die Regel „letzte Runde“ an dieser Stelle ändert."""
    st, r = view.st, view.result
    if view.info is None:
        return "warning", ("In dieser Domäne gibt es kein Bündel, das beiden Seiten mindestens den Reservationsnutzen bringt: eine Einigung ist ausgeschlossen. "
                           "Reservationsnutzen senken oder einen anderen Zufalls-Seed wählen.")
    rule_text = ""
    if view.other.agree != r.agree or (r.agree and view.other.t != r.t):
        a, b = ("an", "aus") if st.rule_on else ("aus", "an")
        rule_text = (f" Mit der Regel „letzte Runde“ {b} wäre es anders gekommen: "
                     + (f"Einigung in Runde {view.other.t}." if view.other.agree else "keine Einigung.") + f" (Eingestellt: {a}.)")
    if not r.agree:
        return "warning", (f"Keine Einigung in {st.T} Runden: {st.first} und {st.second} bekommen beide den Reservationsnutzen {de(float(view.dom.r[0]), 2)} (Summe {de(2 * float(view.dom.r[0]), 2)}), "
                           f"die größte Summe rationaler Bündel wäre {de(view.info['maxsum'], 2)}.{rule_text}")
    e = view.ev
    assert e is not None
    kind = "info"
    last = ", in der letzten Runde" if r.t == st.T else ""
    return kind, (f"Einigung in Runde {r.t} von {st.T}{last}: Verlader ({st.first}) {de(r.u0, 3)}, Spediteur ({st.second}) {de(r.u1, 3)}, Summe {de(e['wel'], 3)} gegenüber {de(view.info['maxsum'], 3)} als "
                  f"größter Summe. Abstand zur Pareto-Grenze {de(e['dist'], 3)}, relatives Nash-Produkt {de(e['nashrel'], 2)}.{rule_text}")


# --------------------------------------------------------------------------------- Erklärtabelle: erste Gebote
FIRST_BID_ROUNDS = (1, 3, 5, 7)


def first_bid_rows(st: Settings) -> list[dict]:
    """Die Gebote jeder Strategie des Auswahlfelds als Verlader gegen einen Hartnäckigen (nimmt nie an, außer durch die Regel „letzte Runde“) in der Domäne und mit der Frist der Seitenleiste:
    Nutzen des eigenen Gebots in den Runden 1, 3, 5, 7 und im letzten eigenen Gebot. So zeigt jede Zeile den Weg des Ziels, ohne dass der Gegner antwortet."""
    dom = build_domain(st)
    rows = []
    for name in ST.POOL:
        trace: list = []
        NG.negotiate(dom, NG.make_agent(name), NG.make_agent(ST.HARTNAECKIG), st.T, st.rule_on, st.band, trace)
        own = {t: float(dom.u[0][o]) for t, p, o, _, _ in trace if p == 0}
        last = max(own) if own else None
        row = {"Strategie": short(name)}
        for rnd in FIRST_BID_ROUNDS:
            row[f"Runde {rnd}"] = de(own[rnd], 3) if rnd in own else "–"
        row["letztes eigenes Gebot"] = f"{de(own[last], 3)} (Runde {last})" if last is not None else "–"
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------- Live: Gegnermodell entlang der Aushandlung
def model_trace(view: LiveView) -> list[dict]:
    """Was ein Häufigkeitsmodell der jeweils antwortenden Seite nach den bisherigen Angeboten des Gegners geschätzt hätte (auch wenn die Strategie keins nutzt): je Seite und Zahl k der gesehenen Angebote
    der Gewichtsabstand und die Rangkorrelation von geschätztem und wahrem Gegnernutzen."""
    out = []
    for p in (0, 1):                       # Beobachter p sieht die Angebote von 1 − p
        offers = [o for _, q, o, _, _ in view.trace if q != p]
        for k in range(1, len(offers) + 1):
            obs = NG.Agent(ST.Spec(curve="zeit", beta=1.0, ac="never", model="haeufig"))
            obs.reset(view.dom, p, view.st.T, False)
            for o in offers[:k]:
                obs.observe(o)
            out.append({"Beobachter": "Verlader" if p == 0 else "Spediteur", "k": k, "gap": OPP.weight_error(obs.est_weights(), view.dom.w[1 - p]),
                        "rho": OPP.rank_correlation(obs.est(), view.dom.u[1 - p])})
    return out


def verdict_model_live(view: LiveView, rows: list[dict]) -> tuple[str, str]:
    if not rows:
        return "info", "Es gab keine Angebote des Gegners zu beobachten."
    guess = OPP.weight_error(np.ones(view.dom.n) / view.dom.n, view.dom.w[1])
    parts = []
    for who in ("Verlader", "Spediteur"):
        sub = [r for r in rows if r["Beobachter"] == who]
        if sub:
            last = sub[-1]
            parts.append(f"Als {who} nach {last['k']} Angeboten des Gegners: Gewichtsabstand {de(last['gap'], 3)}, Rangkorrelation {signed(last['rho'], 3)}")
    return "info", (" · ".join(parts) + f". Zum Vergleich: gleiche Gewichte raten gäbe für den Spediteur in dieser Domäne einen Abstand von {de(guess, 3)}. "
                    f"Eine Aushandlung ist eine Stichprobe; die Reihen unten zeigen das Mittel über {len(C.SEEDS) * C.MODEL_DOMAINS} Domänen.")


# --------------------------------------------------------------------------------- E1: Domäne und Orakel (live Stichprobe)
def domain_stats(count: int, s_values=(0.0, 0.25, 0.5, 0.75, 1.0), g: float = 0.5, n: int = 4, m: int = 4, r: float = 0.2, tag: str = "dstat") -> list[dict]:
    """Je Gewichtsunterschied `count` Domänen (feste Seeds aus der Bezeichnung): Gewichtsabstand, Zahl der Bündel auf der Grenze der rationalen Bündel, Nash-Summe, größte Summe, Nutzen im Nash-Bündel."""
    rows = []
    for s in s_values:
        doms, skipped = D.domain_set(f"{tag}|{s:g}|{g:g}|{n}|{m}|{r:g}", count, s=s, g=g, n=n, m=m, r=r)
        gap, front, nsum, msum, n0, n1 = [], [], [], [], [], []
        for d, info in doms:
            nb = info["nash"]
            gap.append(D.weight_gap(d))
            front.append(len(info["front"]))
            nsum.append(d.u[0][nb] + d.u[1][nb])
            msum.append(info["maxsum"])
            n0.append(d.u[0][nb])
            n1.append(d.u[1][nb])
        rows.append({"s": s, "g": g, "skipped": skipped, "gap": float(np.mean(gap)), "front": float(np.mean(front)), "nash_sum": float(np.mean(nsum)), "max_sum": float(np.mean(msum)),
                     "nash0": float(np.mean(n0)), "nash1": float(np.mean(n1))})
    return rows


def domain_stats_rows(rows: list[dict]) -> list[dict]:
    return [{"Gewichtsunterschied s": de(r["s"], 2), "Gegensatz g": de(r["g"], 2), "Gewichtsabstand": de(r["gap"], 3), "Bündel auf der Grenze": de(r["front"], 1), "Summe im Nash-Bündel": de(r["nash_sum"], 3),
             "größte Summe": de(r["max_sum"], 3)} for r in rows]


def verdict_domain(rows: list[dict], count: int) -> tuple[str, str]:
    lo, hi = rows[0], rows[-1]
    return "info", (f"Je {thousands(count)} Zufallsdomänen: die Summe der Nutzen im Nash-Bündel steigt von {de(lo['nash_sum'], 3)} bei gleichen Gewichten (s = {de(lo['s'], 1)}) auf {de(hi['nash_sum'], 3)} bei "
                    f"unabhängigen (s = {de(hi['s'], 1)}); die größte Summe von {de(lo['max_sum'], 3)} auf {de(hi['max_sum'], 3)}. Je verschiedener die Seiten gewichten, desto mehr ist zu holen. "
                    f"Die Summe im Nash-Bündel liegt in jeder Zeile höchstens {de(max(r['max_sum'] - r['nash_sum'] for r in rows), 3)} unter der größten Summe: die Nash-Lösung ist hier ein guter Maßstab, aber kein Ergebnis der Strategien.")


# --------------------------------------------------------------------------------- E2: Turnier
def turnier_table(c: R.CellStats, with_ranks: bool = True) -> list[dict]:
    """Zeilen nach eigenem Nutzen absteigend: eigener Nutzen (Std), Summe, Abstand zur Grenze, Einigungsrate, Nash-Produkt relativ, Rang je Seed."""
    rows = []
    for i in np.argsort(-c.mean["own"]):
        row = {"Strategie": short(c.names[i]), "Eigener Nutzen (Std)": pm(c.mean["own"][i], c.sd["own"][i]), "Summe": de(c.mean["wel"][i], 3), "Abstand zur Grenze": de(c.mean["dist"][i], 3),
               "Einigungsrate": de(c.mean["agree"][i], 3), "Nash-Produkt rel.": de(c.mean["nashrel"][i], 3)}
        if with_ranks:
            row["Rang je Seed"] = ", ".join(str(int(r)) for r in c.ranks[:, i])
        rows.append(row)
    return rows


def winners_table(data: dict) -> list[dict]:
    w = R.winners(data)
    rows = []
    for T in C.TURNIER_TS:
        row = {"Frist T": str(T)}
        for s in C.TURNIER_SS:
            name, k = w[(T, s)]
            row[f"s = {de(s, 1)}"] = f"{short(name)} ({k} von {len(C.SEEDS)})"
        rows.append(row)
    return rows


def top_group_facts(data: dict) -> dict:
    """Boulware + Modell gegen Boulware über die neun Zellen: in wie vielen Zellen vorn (im Mittel), kleinster und größter Vorsprung (eigener Nutzen), dazu die Zelle (T = 20, s = 0,5)."""
    diffs = {}
    for T in C.TURNIER_TS:
        for s in C.TURNIER_SS:
            c = R.turnier_cell(data, T, s)
            diffs[(T, s)] = float(c.mean["own"][P6.index(TOP)] - c.mean["own"][P6.index(ST.BOULWARE)])
    ahead = [k for k, v in diffs.items() if v > 0]
    return {"diffs": diffs, "ahead": len(ahead), "min": min(diffs.values()), "max": max(diffs.values()), "behind": [k for k, v in diffs.items() if v <= 0]}


def pair_rows(c: R.CellStats, metric: str = "u0") -> list[dict]:
    """Paarmatrix als Zeilen: Zeile = zuerst bietende Strategie, Spalte = zweite; Wert = Nutzen der ersten (u0), der zweiten (u1) bzw. Einigungsrate."""
    M = R.pair_matrix(c, metric)
    return [{"zuerst \\ als Zweite": short(c.names[i]), **{short(c.names[j]): de(M[i, j], 3) for j in range(len(c.names))}} for i in range(len(c.names))]


def verdict_turnier(data: dict) -> tuple[str, str]:
    last_mean, last_seed, exc = R.last_place_counts(data)
    w = R.winners(data)
    c = R.turnier_cell(data, 20, 0.5)
    f = top_group_facts(data)
    tft6 = sum(1 for s in C.TURNIER_SS if w[(6, s)][0] == ST.TFT)
    i_top, i_b, i_t = (P6.index(x) for x in (TOP, ST.BOULWARE, ST.TFT))
    spread = float(max(c.mean["own"][[i_top, i_b, i_t]]) - min(c.mean["own"][[i_top, i_b, i_t]]))
    return "info", (
        f"Conceder liegt in {last_mean} von {len(C.TURNIER_TS) * len(C.TURNIER_SS)} Zellen im Mittel auf dem letzten Rang und in {last_seed} von {len(C.TURNIER_TS) * len(C.TURNIER_SS) * len(C.SEEDS)} Zelle-Seed-Paaren. "
        f"Bei kurzer Frist (T = 6) führt Tit-for-Tat in {tft6} von {len(C.TURNIER_SS)} Gewichtsunterschieden. In der Zelle ({stage_label(20, 0.5)}) liegen drei Strategien vorn: "
        f"Boulware + Modell {de(c.mean['own'][i_top], 3)}, Boulware {de(c.mean['own'][i_b], 3)}, Tit-for-Tat {de(c.mean['own'][i_t], 3)} (zusammen innerhalb von {de(spread, 3)}, Standardabweichung der Seeds {de(max(c.sd['own'][[i_top, i_b, i_t]]), 3)} und kleiner). "
        f"Boulware + Modell liegt in {f['ahead']} von 9 Zellen vor Boulware, mit einem Vorsprung zwischen {de(f['min'], 3)} und {de(f['max'], 3)}: in dieser Zelle ist die Spitze eine Gruppe, bei kurzer Frist führt Tit-for-Tat allein.")


# --------------------------------------------------------------------------------- E3: Position, Regel, Endspiel
def position_table(data: dict, rule: bool = True, opps=(ST.BOULWARE, ST.TFT, ST.HARTNAECKIG), ts=C.POSITION_TS) -> list[dict]:
    rows = []
    for T in ts:
        for pos in (0, 1):
            for opp in opps:
                c = R.position_cell(data, T, rule, pos, opp)
                rows.append({"Frist T": str(T), "Hartnäckig bietet": ("zuerst" if pos == 0 else "als Zweite") + (", zuletzt" if c["last"] else ", nicht zuletzt"), "Gegner": short(opp),
                             "Nutzen Hartnäckig (Std)": pm(c["mine"], c["mine_sd"]), "Nutzen Gegner": de(c["theirs"], 3), "Einigungsrate": de(c["agree"], 3),
                             "Einigungsrunde": "–" if c["round"] != c["round"] else de(c["round"], 1)})
    return rows


def position_facts(data: dict) -> dict:
    """Hartnäckig gegen Boulware bei T = 20 und 21, Regel an/aus, erste/zweite Position."""
    g = lambda T, rule, pos: R.position_cell(data, T, rule, pos, ST.BOULWARE)   # noqa: E731
    return {"t20_first": g(20, True, 0), "t20_second": g(20, True, 1), "t21_first": g(21, True, 0), "t21_second": g(21, True, 1), "t20_first_off": g(20, False, 0), "t20_second_off": g(20, False, 1)}


def verdict_position(data: dict) -> tuple[str, str]:
    f = position_facts(data)
    return "info", (
        f"Hartnäckig gegen Boulware, Regel „letzte Runde“ an: als erste Position (bei T = 20 bietet die zweite zuletzt) {de(f['t20_first']['mine'], 3)} (Std {de(f['t20_first']['mine_sd'], 3)}, Einigungsrate {de(f['t20_first']['agree'], 3)}), "
        f"als zweite Position {de(f['t20_second']['mine'], 3)} (Einigungsrate {de(f['t20_second']['agree'], 3)}); bei T = 21, wo die erste zuletzt bietet, kehrt es sich um: "
        f"{de(f['t21_first']['mine'], 3)} als erste, {de(f['t21_second']['mine'], 3)} als zweite. Ohne die Regel: {de(f['t20_first_off']['mine'], 3)} und {de(f['t20_second_off']['mine'], 3)}. "
        "Gegen Boulware (ebenso gegen Linear und Conceder, gemessen in der Standarddomäne) zahlt sich Hartnäckigkeit also nur aus, wenn der Gegner zuletzt bietet und die Regel „letzte Runde“ gilt: der Vorteil des ersten Gebots ist hier ein Vorteil des nicht zuletzt Bietenden.")


def rule_rows(data: dict) -> list[dict]:
    """Wirkungstabelle der Regel „letzte Runde“ im Pool-Turnier (T = 20, s = 0,5, 100 Domänen je Seed): A = Regel an, B = Regel aus."""
    on, off = R.turnier_cell(data, 20, 0.5), R.rule_off_cell(data)
    r_on, r_off = TO.ranks_desc(on.mean["own"]), TO.ranks_desc(off.mean["own"])
    rows = []
    for i in np.argsort(-on.mean["own"]):
        rows.append({"Strategie": short(P6[i]), "A: Regel an, eigener Nutzen": de(on.mean["own"][i], 3), "B: Regel aus, eigener Nutzen": de(off.mean["own"][i], 3), "Rang A / B": f"{r_on[i]} / {r_off[i]}",
                     "Einigungsrate A / B": f"{de(on.mean['agree'][i], 3)} / {de(off.mean['agree'][i], 3)}"})
    return rows


def verdict_rule(data: dict) -> tuple[str, str]:
    on, off = R.turnier_cell(data, 20, 0.5), R.rule_off_cell(data)
    h = P6.index(ST.HARTNAECKIG)
    rest = [i for i in range(len(P6)) if i != h]
    shift = max(abs(on.mean["own"][i] - off.mean["own"][i]) for i in rest)
    same_order = list(np.argsort(-on.mean["own"][rest])) == list(np.argsort(-off.mean["own"][rest]))
    return "info", (
        f"Ohne die Regel „letzte Runde“ fällt Hartnäckig von {de(on.mean['own'][h], 3)} auf {de(off.mean['own'][h], 3)} (Einigungsrate {de(on.mean['agree'][h], 3)} auf {de(off.mean['agree'][h], 3)}) und "
        f"von Rang {TO.ranks_desc(on.mean['own'])[h]} auf Rang {TO.ranks_desc(off.mean['own'])[h]}; der eigene Nutzen der fünf übrigen ändert sich um höchstens {de(shift, 3)}, "
        f"{'ihre Rangfolge bleibt' if same_order else 'ihre Rangfolge ändert sich'}. Die Regel ist eine gewählte Rechenregel: beide Fassungen stehen zur Wahl.")


def endgame_rows(c: R.CellStats) -> list[dict]:
    rows = []
    for i in np.argsort(-c.mean["own"]):
        rows.append({"Strategie": short(c.names[i]), "Eigener Nutzen (Std)": pm(c.mean["own"][i], c.sd["own"][i]), "Summe": de(c.mean["wel"][i], 3), "Einigungsrate": de(c.mean["agree"][i], 3),
                     "Rang je Seed": ", ".join(str(int(r)) for r in c.ranks[:, i])})
    return rows


def verdict_endgame(data: dict) -> tuple[str, str]:
    c7, c9, c6 = R.endgame_cell(data, 20, "7"), R.endgame_cell(data, 20, "9"), R.endgame_cell(data, 20, "6")
    e, m, b = (c7.names.index(x) for x in (ST.ENDSPIEL, ST.BOULWARE_MODELL, ST.BOULWARE))
    e9, m9, b9 = (c9.names.index(x) for x in (ST.ENDSPIEL, ST.BOULWARE_MODELL, ST.BOULWARE))
    return "info", (
        f"Pool aus den sechs Strategien und dem Endspiel-Ausnutzer (T = 20, s = 0,5, 4 × 50 Domänen): Endspiel-Ausnutzer {de(c7.mean['own'][e], 3)} gegen {de(c7.mean['own'][m], 3)} für Boulware + Modell ohne Endspiel, "
        f"Rang 1 in {int((c7.ranks[:, e] == 1).sum())} von {len(C.SEEDS)} Seeds; die Einigungsrate fällt dabei von {de(c7.mean['agree'][m], 3)} auf {de(c7.mean['agree'][e], 3)}. "
        f"Im Pool mit den beiden Orakel-Agenten liegt Boulware ({de(c9.mean['own'][b9], 3)}) vor Boulware + Modell ({de(c9.mean['own'][m9], 3)}), in den sechs Standardstrategien mit denselben 50 Domänen je Seed umgekehrt "
        f"({de(c6.mean['own'][P6.index(ST.BOULWARE)], 3)} gegen {de(c6.mean['own'][P6.index(TOP)], 3)}): die Reihenfolge innerhalb der Spitzengruppe hängt am Pool.")


# --------------------------------------------------------------------------------- E4: Gegnermodell
def model_rows(data: dict, s: float = 0.5) -> list[dict]:
    rows = []
    for opp in R.EFFECT_OPPS:
        m = R.model_curve(data, opp, s)
        ks = m["k"]
        rows.append({"Gegner": short(opp), **{f"Gewichtsabstand k = {k}": de(m["gap"][ks.index(k)], 3) for k in (1, 8, 20)}, **{f"Rangkorrelation k = {k}": signed(m["rho"][ks.index(k)], 3) for k in (1, 8, 20)}})
    return rows


def effect_rows(data: dict) -> list[dict]:
    rows = []
    for curve in ("Boulware", "Linear", "Conceder"):
        for model, lab in (("keins", "kein Modell"), ("haeufig", "Häufigkeitsmodell"), ("orakel", "Orakel")):
            c = R.effect_cell(data, curve, model)
            rows.append({"Programm": f"{curve} + {lab}", "Eigener Nutzen": pm(*c["own"]), "Summe": pm(*c["wel"]), "Abstand zur Grenze": pm(*c["dist"]), "Einigungsrate": de(c["agree"][0], 3)})
    return rows


def verdict_model(data: dict) -> tuple[str, str]:
    lin = R.model_curve(data, ST.LINEAR, 0.5)
    con = R.model_curve(data, ST.CONCEDER, 0.5)
    ks = lin["k"]
    i8, i20 = ks.index(8), ks.index(20)
    closed = R.effect_gap_closed(data, "Boulware", "wel")
    c0, c1, c2 = (R.effect_cell(data, "Conceder", m)["own"][0] for m in ("keins", "haeufig", "orakel"))
    return "info", (
        f"Das Häufigkeitsmodell verbessert die Schätzung der Gewichte nur wenig: gegen einen linearen Gegner ist der Gewichtsabstand nach 20 Angeboten {de(lin['gap'][i20], 3)} gegenüber {de(lin['guess'], 3)} beim bloßen Raten gleicher Gewichte "
        f"(s = 0,5, 4 × 100 Domänen). Die Rangkorrelation steigt gegen Linear auf {signed(lin['rho'][i8], 3)} nach 8 Angeboten und fällt auf {signed(lin['rho'][i20], 3)} nach 20; "
        f"gegen Conceder wird sie {signed(con['rho'][i20], 3)}: der Gegner kommt auf mich zu, das Modell lernt meine Vorlieben statt seiner, es wird getäuscht. "
        f"Bei Boulware schließt das Modell {pct(closed, 0)} der Summenlücke zum Orakel; der eigene Nutzen von Conceder sinkt mit Modell ({de(c0, 3)} → {de(c1, 3)} → {de(c2, 3)} mit Orakel), die Summe steigt.")


# --------------------------------------------------------------------------------- E5: Zerlegung
def eta_rows(data: dict, stage_set: str) -> list[dict]:
    arrs, combos = R.decomp_arrays(data), R.decomp_combos()
    rows = []
    for m, lab in R.DECOMP_METRICS.items():
        e = R.eta_squared(arrs[m], combos, R.STAGE_SETS[stage_set])
        rows.append({"Kennzahl": lab, **{f: pm(*e[f]) for f in (*R.FACTORS, "Rest")}})
    return rows


def eta_values(data: dict, stage_set: str, metric: str) -> dict:
    return R.eta_squared(R.decomp_arrays(data)[metric], R.decomp_combos(), R.STAGE_SETS[stage_set])


def accept_rows(data: dict) -> list[dict]:
    return [{"Annahmeregel (Kurve Linear, kein Modell)": r["label"], "Eigener Nutzen": pm(*r["own"]), "Summe": pm(*r["wel"]), "Einigungsrate": de(r["agree"][0], 3)} for r in R.accept_cells(data)]


def verdict_decomp(data: dict) -> tuple[str, str]:
    e_own, e_wel = eta_values(data, "alle 48 Agenten", "own"), eta_values(data, "alle 48 Agenten", "wel")
    o_res = eta_values(data, "ohne „ab Reservation“", "own")
    o_both = eta_values(data, "ohne beides (27 Agenten)", "wel")
    return "info", (
        f"Varianzanteil der 48 Agentenmittel, eigener Nutzen: Akzeptanz {de(e_own['Akzeptanz'][0], 3)}, Kurve {de(e_own['Kurve'][0], 3)}, Modell {de(e_own['Modell'][0], 3)}; Summe der Nutzen: Kurve {de(e_wel['Kurve'][0], 3)}, "
        f"Modell {de(e_wel['Modell'][0], 3)}, Akzeptanz {de(e_wel['Akzeptanz'][0], 3)}. Die Anteile hängen an den gewählten Stufen: ohne die Stufe „ab Reservation“ steigt der Anteil der Kurve am eigenen Nutzen auf "
        f"{de(o_res['Kurve'][0], 3)} (Akzeptanz {de(o_res['Akzeptanz'][0], 3)}); ohne diese Stufe und ohne Tit-for-Tat hat das Modell {de(o_both['Modell'][0], 3)} der Varianz der Summe. "
        "Eine feste Rangfolge Gebot vor Annahme vor Modell ergibt sich in diesem Modell nicht.")


# --------------------------------------------------------------------------------- E6: Robustheit
def robust_rows(data: dict) -> list[dict]:
    rows = []
    for r in R.robust_cells(data):
        rows.append({"Änderung (alles andere wie in der Basis)": r["label"], "Sieger": short(r["top"]), "Seeds mit diesem Sieger": f"{r['wins']} von {len(C.SEEDS)}", "Eigener Nutzen des Siegers": de(r["top_own"], 3),
                     "Einigung des Siegers": de(r["top_agree"], 3), "Letzter": short(r["last"])})
    return rows


def robust_counts(data: dict) -> dict:
    cells = R.robust_cells(data)
    return {"n": len(cells), "conceder_last": sum(c["last"] == ST.CONCEDER for c in cells), "not_last": [c["label"] for c in cells if c["last"] != ST.CONCEDER],
            "winners": {name: [c["label"] for c in cells if c["top"] == name] for name in dict.fromkeys(c["top"] for c in cells)}}


def verdict_robust(data: dict) -> tuple[str, str]:
    rc = robust_counts(data)
    parts = "; ".join(f"{short(n)} in {len(v)}" for n, v in rc["winners"].items())
    return "info", (f"Bei {rc['n']} Reglerwerten (je eine Änderung gegenüber der Basis, 4 × 50 Domänen) gewinnt {parts}. Conceder ist in {rc['conceder_last']} von {rc['n']} Fällen Letzter"
                    + (f" (nicht bei {', '.join(rc['not_last'])})" if rc["not_last"] else "") + ". Der Sieger der Spitzengruppe ist also ein Ergebnis der Einstellungen, nicht der Strategie allein.")


# --------------------------------------------------------------------------------- E7: Lernender
def learner_rows(data: dict) -> list[dict]:
    rows = []
    for E in C.LEARN_EPISODES:
        s = R.learner_summary(data, E)
        rows.append({"Training": f"{thousands(E)} Episoden", "Mittel über 6 Gegner (Std)": pm(s["mean"], s["sd"]), "Spanne der vier Trainings-Seeds": f"{de(s['min'], 3)} bis {de(s['max'], 3)}", "Einigungsrate": de(s["agree"], 3)})
    b = R.learner_baselines(data)
    for name in (ST.BOULWARE_MODELL, ST.BOULWARE, ST.LINEAR, ST.CONCEDER):
        rows.append({"Training": short(name) + " (fest)", "Mittel über 6 Gegner (Std)": pm(b[name]["mean"], b[name]["sd"]), "Spanne der vier Trainings-Seeds": "–", "Einigungsrate": "–"})
    return rows


def learner_opponent_rows(data: dict) -> list[dict]:
    s = R.learner_summary(data, C.LEARN_REF_EPISODES)
    b = R.learner_baselines(data)
    return [{"Gegner": short(o), "Lernender (Std)": pm(*s["per_opp"][o]), "Boulware": de(b[ST.BOULWARE]["per_opp"][o], 3), "Boulware + Modell": de(b[ST.BOULWARE_MODELL]["per_opp"][o], 3),
             "Linear": de(b[ST.LINEAR]["per_opp"][o], 3), "Conceder": de(b[ST.CONCEDER]["per_opp"][o], 3)} for o in P6]


def transfer_rows(data: dict) -> list[dict]:
    return [{"Bedingung (Training bei T = 20, s = 0,5, Regel an)": r["label"], "Lernender (Std)": pm(r["lern"], r["lern_sd"]), "Boulware + Modell": de(r["modell"], 3), "Boulware": de(r["boulware"], 3), "Linear": de(r["linear"], 3),
             "Trainings-Seeds vor beiden": f"{r['ahead']} von {len(C.SEEDS)}"} for r in R.learner_transfer(data)]


def omitted_rows(data: dict) -> list[dict]:
    rows = []
    for r in R.learner_omitted(data):
        rows.append({"Im Training ausgelassen": short(r["left"]), "Lernender gegen ihn (Std)": pm(r["mean"], r["sd"]), "Lernender gegen die vier übrigen": de(r["others"], 3),
                     "Boulware / Linear / Conceder gegen ihn": " / ".join(de(r["fixed"][n], 3) for n in (ST.BOULWARE, ST.LINEAR, ST.CONCEDER))})
    return rows


def verdict_learner(data: dict) -> tuple[str, str]:
    s5, s20, s50 = (R.learner_summary(data, E) for E in C.LEARN_EPISODES)
    b = R.learner_baselines(data)
    tr = R.learner_transfer(data)
    worst = min(tr, key=lambda r: r["ahead"])
    gaps = {o: s20["per_opp"][o][0] - b[ST.BOULWARE_MODELL]["per_opp"][o] for o in P6}
    big = max(gaps, key=gaps.get)
    return "warning", (
        f"Der Lernende liegt nach {thousands(C.LEARN_EPISODES[0])} / {thousands(C.LEARN_EPISODES[1])} / {thousands(C.LEARN_EPISODES[2])} Episoden bei {de(s5['mean'], 3)} / {de(s20['mean'], 3)} / {de(s50['mean'], 3)} "
        f"(Std über die Trainings-Seeds {de(s5['sd'], 3)} / {de(s20['sd'], 3)} / {de(s50['sd'], 3)}) gegen Boulware + Modell mit {de(b[ST.BOULWARE_MODELL]['mean'], 3)}; mehr Training hilft nicht durchgehend. "
        f"Das gilt gegen diesen Pool und diesen Domänengenerator: bei der schwächsten Übertragungsbedingung ({worst['label']}) liegen {worst['ahead']} von {len(C.SEEDS)} Trainings-Seeds vor beiden festen Strategien. "
        f"Nach 20.000 Episoden ist der größte Vorsprung vor Boulware + Modell gegen {short(big)} ({de(s20['per_opp'][big][0], 3)} gegen {de(b[ST.BOULWARE_MODELL]['per_opp'][big], 3)}), "
        f"gegen {sum(g < 0 for g in gaps.values())} Gegner liegt der Lernende dahinter; über andere Gegner sagt die Messung nichts.")
