"""Jede Zahl aus der README und den Hilfetexten, nachgerechnet (DEMO-PLAYBOOK §4, §11a).

Drei Quellen, alle plattformsicher:
* die gespeicherten Messreihen (`data/messreihen.json`: vier Zufalls-Seeds je Zelle, Erzeuger-Hash geprüft in `test_measurements.py`): die README-Zahl ist der gespeicherte Wert, gerundet;
* Rechnung im Test (Zählungen über Zellen, Gegenprobe mit anderen Seeds, Orakel-Zählungen aus den anderen Testdateien);
* die Zeilenbauer der App (`vha_evaluation.*_table`, `*_rows`): jede Zeile der README-Tabellen steht als Text in der README.
Keine Zeit-Assertion, keine exakten Zufallswerte einzelner Läufe. Zählungen über das Raster („in k von m Zellen“) stehen als Zählungen im Test, nicht als Urteilswort."""

import re
from pathlib import Path

import numpy as np
import pytest

import vha_constants as C
import vha_domain as D
import vha_evaluation as V
import vha_experiments as XP
import vha_negotiation as NG
import vha_results as R
import vha_strategies as ST
import vha_texts as T
import vha_tournament as TO

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
de = V.de


def row_of(label):
    """Die Tabellenzeile der Hypothese (eine Zeile der README, beginnt mit `| Hn |`)."""
    return next(line for line in README.splitlines() if line.startswith(f"| {label} |"))


def section(start, end):
    return README.split(start)[1].split(end)[0]


def has(text, *needles):
    missing = [n for n in needles if n not in text]
    assert not missing, missing


@pytest.fixture(scope="module")
def data():
    d = XP.load()
    assert d is not None
    return d


@pytest.fixture(scope="module")
def dstat():
    """Die Domänenstatistik der README (400 Domänen je Zeile), einmal gerechnet."""
    return V.domain_stats(400)


def _md_row(d):
    return "| " + " | ".join(str(v) for v in d.values()) + " |"


def _md_header(rows):
    return "| " + " | ".join(rows[0]) + " |"


# --- README und App: dieselben Texte, dieselben Tabellenzeilen ---------------------------------------------------------------------------------
def test_assumption_table_is_the_same_text_in_app_and_readme():
    assert T.ASSUMPTIONS_MARKDOWN.strip() in README


def test_readme_tables_are_built_by_the_same_row_builders(data, dstat):
    tables = [V.domain_stats_rows(dstat), V.winners_table(data)]
    tables += [V.turnier_table(R.turnier_cell(data, t, 0.5)) for t in (6, 20, 60)]
    tables += [V.position_table(data, True, ts=(20, 21)), V.position_table(data, False, ts=(20, 21)), V.rule_rows(data), V.endgame_rows(R.endgame_cell(data, 20, "7")),
               V.endgame_rows(R.endgame_cell(data, 20, "9")), V.model_rows(data, 0.5), V.effect_rows(data)]
    tables += [V.eta_rows(data, name) for name in R.STAGE_SETS]
    tables += [V.accept_rows(data), V.robust_rows(data), V.learner_rows(data), V.learner_opponent_rows(data), V.transfer_rows(data), V.omitted_rows(data)]
    for rows in tables:
        assert _md_header(rows) in README, rows[0]
        for row in rows:
            assert _md_row(row) in README, row


def test_hypothesis_table_has_h1_to_h13_with_a_clear_stand():
    rows = re.findall(r"^\| (H\d+) \|(?:\\\||[^|])*\|\s*\*\*([^*]+)\*\*", README, flags=re.M)
    assert [h for h, _ in rows] == [f"H{i}" for i in range(1, 14)]
    stands = [a.split(",")[0].split(" ")[0] for _, a in rows]
    assert stands == ["Ja", "Ja", "Ja", "Nur", "Ja", "Nein", "Ja", "Nein", "Ja", "Ja", "Ja", "Ja", "Ja"]


# --- Kernfrage -------------------------------------------------------------------------------------------------------------------------------
def test_core_question_numbers(data):
    core = section("## Kernfrage", "## Befunde und Korrekturen")
    last_mean, last_seed, _ = R.last_place_counts(data)
    c = R.turnier_cell(data, 20, 0.5)
    f = V.top_group_facts(data)
    p = V.position_facts(data)
    lin, con = R.model_curve(data, ST.LINEAR, 0.5), R.model_curve(data, ST.CONCEDER, 0.5)
    e_own, e_wel = V.eta_values(data, "alle 48 Agenten", "own"), V.eta_values(data, "alle 48 Agenten", "wel")
    o_res = V.eta_values(data, "ohne „ab Reservation“", "own")
    s20, s50 = R.learner_summary(data, 20000), R.learner_summary(data, 50000)
    b = R.learner_baselines(data)
    assert last_mean == 9 and "in 9 von 9 Zellen" in core
    top3 = c.mean["own"][[5, 1, 4]]
    has(core, f"Boulware + Modell ({de(c.mean['own'][5], 3)}), Boulware ({de(c.mean['own'][1], 3)}) und Tit-for-Tat ({de(c.mean['own'][4], 3)}) innerhalb von {de(max(top3) - min(top3), 3)}", f"nur in {f['ahead']} von 9 Zellen vor Boulware")
    assert de(max(top3) - min(top3), 3) == "0,016"
    assert all(R.winners(data)[(6, s)][0] == ST.TFT for s in C.TURNIER_SS) and "bei T = 6 führt Tit-for-Tat in allen drei Gewichtsunterschieden" in core
    has(core, f"{de(p['t20_first']['mine'], 3)} als erste Position gegen {de(p['t20_second']['mine'], 3)} als zweite", f"({de(p['t21_first']['mine'], 3)} gegen {de(p['t21_second']['mine'], 3)})")
    has(core, f"ist der Gewichtsabstand {de(lin['gap'][-1], 3)} (Raten gleicher Gewichte: {de(lin['guess'], 3)})", f"Rangkorrelation {de(con['rho'][-1], 3)}", f"{de(100 * R.effect_gap_closed(data, 'Boulware', 'wel'), 1)} % der Summenlücke")
    has(core, f"die Annahme {de(e_own['Akzeptanz'][0], 3)} der Varianz, die Kurve {de(e_own['Kurve'][0], 3)}, das Modell {de(e_own['Modell'][0], 3)}", f"die Kurve {de(e_wel['Kurve'][0], 3)} und das Modell {de(e_wel['Modell'][0], 3)}",
        f"Anteil der Kurve am eigenen Nutzen auf {de(o_res['Kurve'][0], 3)}")
    has(core, f"({de(s20['mean'], 3)} gegen {de(b[ST.BOULWARE_MODELL]['mean'], 3)} für Boulware + Modell nach 20.000 Episoden)", f"50.000 Episoden: {de(s50['mean'], 3)}")


def test_the_gap_between_two_top_strategies_is_within_the_seed_scatter_in_7_of_9_cells(data):
    f = V.top_group_facts(data)
    within = 0
    for T_ in C.TURNIER_TS:
        for s in C.TURNIER_SS:
            c = R.turnier_cell(data, T_, s)
            within += abs(f["diffs"][(T_, s)]) < max(c.sd["own"][5], c.sd["own"][1])
    assert within == 7 and f["ahead"] == 4 and f["behind"] == [(6, 0.0), (6, 0.5), (20, 0.0), (60, 0.0), (60, 0.5)]
    assert de(f["min"], 3) == "−0,004" and de(f["max"], 3) == "0,009" and abs(f["diffs"][(60, 0.5)]) < 2e-4 and max(max(R.turnier_cell(data, a, b).sd["own"][[1, 5]]) for a in C.TURNIER_TS for b in C.TURNIER_SS) < 0.0075
    has(section("## Befunde und Korrekturen", "## Modell"), "in 4 von 9 Zellen vor Boulware", "von −0,004 bis +0,009", "in 7 von 9 Zellen kleiner als die größere der beiden Standardabweichungen", "(höchstens 0,007)", "Gleichstand (−0,0001)", "(2 von 4 Seeds)")


def test_corrections_section_numbers(data):
    sec = section("## Befunde und Korrekturen", "## Modell")
    s20, s50 = R.learner_summary(data, 20000), R.learner_summary(data, 50000)
    assert de(s50["mean"], 3) == "0,640" and de(s20["mean"], 3) == "0,657" and de(s50["sd"], 3) == "0,015" and de(s20["sd"], 3) == "0,005"
    has(sec, "(0,640 gegen 0,657, Std 0,015 gegen 0,005)", "3 von 4 Trainings-Seeds")
    assert [r["ahead"] for r in R.learner_transfer(data) if r["label"] == "s = 1"] == [3]
    assert R.winners(data)[(60, 0.5)] == (ST.BOULWARE, 2)
    assert D.dom_info(D.make_domain(1)) is not None and "400 Domänen" in sec and "alle 360 Ecken" in sec


# --- H1 bis H3 --------------------------------------------------------------------------------------------------------------------------------
def test_h1_conceder_is_last(data):
    last_mean, last_seed, exc = R.last_place_counts(data)
    rc = V.robust_counts(data)
    assert (last_mean, last_seed, exc) == (9, 36, []) and rc["conceder_last"] == 17 and rc["not_last"] == ["Reservationsnutzen 0,5"]
    cells = {c["label"]: c for c in R.robust_cells(data)}
    assert cells["Reservationsnutzen 0,5"]["last"] == ST.HARTNAECKIG
    assert cells["Conceder-Exponent 2"]["last"] == ST.CONCEDER and cells["Conceder-Exponent 10"]["last"] == ST.CONCEDER
    has(README, "auch mit den Exponenten 2 und 10 (Robustheit)")
    has(row_of("H1"), "Letzter in 9 von 9 Zellen", "in 36 von 36 Zelle-Seed-Paaren", "in 17 von 18 Reglerwerten", "Ausnahme Reservationsnutzen 0,5: dort ist Hartnäckig Letzter")


def test_h2_the_top_is_a_group_whose_order_depends_on_the_pool(data):
    c = R.turnier_cell(data, 20, 0.5)
    assert list(np.argsort(-c.mean["own"])[:3]) == [5, 1, 4]
    c9, c6 = R.endgame_cell(data, 20, "9"), R.endgame_cell(data, 20, "6")
    b9, m9 = c9.names.index(ST.BOULWARE), c9.names.index(ST.BOULWARE_MODELL)
    assert c9.mean["own"][b9] > c9.mean["own"][m9] and c6.mean["own"][5] > c6.mean["own"][1]
    has(row_of("H2"), f"Boulware + Modell {de(c.mean['own'][5], 3)}, Boulware {de(c.mean['own'][1], 3)}, Tit-for-Tat {de(c.mean['own'][4], 3)}", "in 4 von 9 Zellen vor Boulware (−0,004 bis +0,009)", "bei T = 60, s = 0,5 gleichauf",
        f"Boulware ({de(c9.mean['own'][b9], 3)}) vor Boulware + Modell ({de(c9.mean['own'][m9], 3)})", f"({de(c6.mean['own'][1], 3)} gegen {de(c6.mean['own'][5], 3)})")


def test_h3_tit_for_tat_wins_at_short_horizons(data):
    w = R.winners(data)
    assert all(w[(6, s)] == (ST.TFT, 4) for s in C.TURNIER_SS) and w[(20, 0.0)] == (ST.TFT, 4)
    assert not any(w[(T_, s)][0] == ST.TFT for T_, s in w if T_ >= 20 and s >= 0.5)
    rob = {c["label"]: c for c in R.robust_cells(data)}
    assert rob["Frist T = 7"]["top"] == ST.TFT and rob["Boulware-Exponent 0,1"]["top"] == ST.TFT and rob["Frist T = 7"]["wins"] == 4
    has(row_of("H3"), "Rang 1 in 3 von 3 Gewichtsunterschieden (je 4 von 4 Seeds)", "bei T = 7 und beim Boulware-Exponenten 0,1", "bei T = 20 und s = 0 ebenfalls (4 von 4)", "ab T = 20 mit s ≥ 0,5 nicht mehr Erster")


# --- H4, H5 -----------------------------------------------------------------------------------------------------------------------------------
def test_h4_stubborn_only_pays_for_the_one_who_does_not_bid_last(data):
    f = V.position_facts(data)
    row = row_of("H4")
    has(row, f"{de(f['t20_first']['mine'], 3)} (Std {de(f['t20_first']['mine_sd'], 3)})", f"{de(f['t20_second']['mine'], 3)} (Std {de(f['t20_second']['mine_sd'], 3)})", f"{de(f['t21_first']['mine'], 3)} gegen {de(f['t21_second']['mine'], 3)}",
        f"{de(f['t20_first_off']['mine'], 3)} und {de(f['t20_second_off']['mine'], 3)}")
    assert f["t20_first"]["mine"] > 0.7 > f["t20_second"]["mine"] and f["t21_second"]["mine"] > 0.7 > f["t21_first"]["mine"]
    rob = {c["label"]: c for c in R.robust_cells(data)}
    assert rob["Reservationsnutzen 0"]["top"] == ST.HARTNAECKIG and rob["Gegensatz g = 0"]["top"] == ST.HARTNAECKIG and rob["Reservationsnutzen 0"]["wins"] == rob["Gegensatz g = 0"]["wins"] == 4
    assert de(rob["Reservationsnutzen 0"]["top_own"], 3) == "0,900" and de(rob["Gegensatz g = 0"]["top_own"], 3) == "0,900"


def test_position_cells_against_linear_and_conceder_equal_those_against_boulware(data):
    """Aussage in README und App: gegen Linear und Conceder dieselben Nutzen und Einigungsraten wie gegen Boulware (nur die Einigungsrunde unterscheidet sich)."""
    differs_in_round = False
    for T_ in C.POSITION_TS:
        for rule in (True, False):
            for pos in (0, 1):
                ref = R.position_cell(data, T_, rule, pos, ST.BOULWARE)
                for opp in (ST.LINEAR, ST.CONCEDER):
                    c = R.position_cell(data, T_, rule, pos, opp)
                    assert (c["mine"], c["theirs"], c["agree"]) == pytest.approx((ref["mine"], ref["theirs"], ref["agree"]), abs=1e-9)
                    differs_in_round |= c["round"] != ref["round"]
    assert differs_in_round
    has(README, "gegen Linear und Conceder ergeben sich in allen Zellen dieselben Nutzen und Einigungsraten wie gegen Boulware, nur die mittlere Einigungsrunde unterscheidet sich")


def test_h5_two_stubborn_ones_fail(data):
    for T_ in C.TURNIER_TS:
        assert de(R.pair_agreement(data, T_, ST.HARTNAECKIG, ST.HARTNAECKIG), 3) == "0,090" and R.pair_agreement(data, T_, ST.TFT, ST.TFT) == pytest.approx(R.pair_agreement(data, T_, ST.HARTNAECKIG, ST.HARTNAECKIG))
    for T_ in C.TURNIER_TS:
        M = R.turnier_matrices(data, T_, 0.5)
        assert np.allclose(M[:, 0, 0], M[:, 4, 4], atol=1e-9)                     # ganze Kennzahlenzeilen sind gleich
    has(row_of("H5"), "Einigungsrate 0,090 bei T = 6, 20 und 60", "zahlengleich")
    import test_branches as TB
    assert len(TB.DOMS) == 40 and "(40 von 40 Domänen)" in README


# --- H6, H7 -----------------------------------------------------------------------------------------------------------------------------------
def test_h6_the_model_learns_weights_slowly_and_is_deceived_by_conceder(data):
    got = {opp: R.model_curve(data, opp, 0.5) for opp in R.EFFECT_OPPS}
    guess = got[ST.LINEAR]["guess"]
    ks = got[ST.LINEAR]["k"]
    i8, i12, i20 = ks.index(8), ks.index(12), ks.index(20)
    row = row_of("H6")
    for opp, label in ((ST.LINEAR, "0,172 (Linear)"), (ST.BOULWARE, "0,197 (Boulware)"), (ST.CONCEDER, "0,205 (Conceder)"), (ST.TFT, "0,210 (Tit-for-Tat)"), (ST.HARTNAECKIG, "0,220 (Hartnäckig)")):
        assert de(got[opp]["gap"][i20], 3) == label.split(" ")[0]
        assert got[opp]["gap"][i20] <= guess + 1e-9
        has(row, label)
    assert de(guess, 3) == "0,220" and de(got[ST.CONCEDER]["rho"][i20], 3) == "−0,398" and de(got[ST.CONCEDER]["rho"][i8], 3) == "0,070" and got[ST.CONCEDER]["rho"][i12] < 0
    has(row, "Rangkorrelation −0,398 nach 20 Angeboten (+0,070 nach 8)")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "Gegen Conceder ist sie bei 12 und bei 20 Angeboten negativ" in app
    assert all(got[ST.CONCEDER]["rho"][ks.index(k)] < 0 for k in (12, 20)) and got[ST.CONCEDER]["rho"][ks.index(8)] > 0


def test_h7_the_model_closes_part_of_the_gap_and_helps_the_own_utility_little(data):
    row = row_of("H7")
    cell = {m: R.effect_cell(data, "Boulware", m) for m in ("keins", "haeufig", "orakel")}
    assert [de(cell[m]["wel"][0], 3) for m in cell] == ["1,034", "1,085", "1,161"] and [de(cell[m]["own"][0], 3) for m in cell] == ["0,551", "0,573", "0,608"]
    assert [de(100 * R.effect_gap_closed(data, c, "wel"), 1) for c in ("Boulware", "Linear", "Conceder")] == ["40,3", "44,2", "47,3"] and de(100 * R.effect_gap_closed(data, "Boulware", "dist"), 1) == "41,9"
    con = [R.effect_cell(data, "Conceder", m)["own"][0] for m in ("keins", "haeufig", "orakel")]
    assert [de(x, 3) for x in con] == ["0,395", "0,389", "0,384"] and con[0] > con[1] > con[2]
    has(row, "40,3 % der Summenlücke (1,034 → 1,085 → 1,161), 41,9 % der Abstandslücke", "eigener Nutzen 0,551 → 0,573 → 0,608", "Linear 44,2 %, Conceder 47,3 %", "(0,395 → 0,389 → 0,384)")
    assert all(R.effect_cell(data, c, m)["agree"][0] > 0.9 for c in ("Boulware", "Linear", "Conceder") for m in ("keins", "haeufig", "orakel"))


# --- H8, H9 -----------------------------------------------------------------------------------------------------------------------------------
def test_h8_the_ranking_of_the_building_blocks_depends_on_metric_and_stages(data):
    own = V.eta_values(data, "alle 48 Agenten", "own")
    wel = V.eta_values(data, "alle 48 Agenten", "wel")
    assert own["Akzeptanz"][0] > own["Kurve"][0] > own["Modell"][0] and wel["Kurve"][0] > wel["Modell"][0] > wel["Akzeptanz"][0]
    o1 = V.eta_values(data, "ohne „ab Reservation“", "own")
    o2 = V.eta_values(data, "ohne beides (27 Agenten)", "wel")
    assert [de(own[k][0], 3) for k in ("Akzeptanz", "Kurve", "Modell")] == ["0,461", "0,402", "0,008"] and [de(wel[k][0], 3) for k in ("Kurve", "Modell", "Akzeptanz")] == ["0,719", "0,186", "0,032"]
    assert de(o1["Kurve"][0], 3) == "0,910" and de(o2["Modell"][0], 3) == "0,752"
    has(row_of("H8"), "Akzeptanz 0,461 > Kurve 0,402 > Modell 0,008", "Kurve 0,719 > Modell 0,186 > Akzeptanz 0,032", "trägt die Kurve 0,910 des eigenen Nutzens", "das Modell 0,752 der Summe")


def test_h9_rank_correlations_and_the_winner_change(data):
    got = {T_: [R.rank_correlation_with_reference(data, T_, s) for s in C.TURNIER_SS] for T_ in C.TURNIER_TS}
    fmt = lambda vals: " / ".join(de(v, 2) for v in vals)           # noqa: E731
    row = row_of("H9")
    has(row, f"T = 6 {fmt(got[6])}", f"T = 20 {fmt(got[20])}", f"T = 60 {fmt(got[60])}")
    w = R.winners(data)
    assert w[(20, 0.0)][0] == ST.TFT and w[(20, 0.5)][0] == ST.BOULWARE_MODELL


# --- H10 --------------------------------------------------------------------------------------------------------------------------------------
def test_h10_the_last_round_rule_shapes_agreement_and_rank_of_the_stubborn(data):
    on, off = R.turnier_cell(data, 20, 0.5), R.rule_off_cell(data)
    r_on, r_off = TO.ranks_desc(on.mean["own"]), TO.ranks_desc(off.mean["own"])
    rest = [i for i in range(6) if i != 0]
    shift = max(abs(on.mean["own"][i] - off.mean["own"][i]) for i in rest)
    assert list(np.argsort(-on.mean["own"][rest])) == list(np.argsort(-off.mean["own"][rest]))
    row = row_of("H10")
    has(row, f"fällt Hartnäckig von {de(on.mean['own'][0], 3)} auf {de(off.mean['own'][0], 3)}", f"(Einigungsrate {de(on.mean['agree'][0], 3)} auf {de(off.mean['agree'][0], 3)}, Rang {r_on[0]} auf {r_off[0]})",
        f"um höchstens {de(shift, 3)}")
    assert (r_on[0], r_off[0]) == (5, 6) and de(shift, 3) == "0,010"
    adv = {T_: R.selfplay_first_advantage(data, T_) for T_ in C.TURNIER_TS}
    has(row, f"bei T = 6 {V.signed(adv[6]['Boulware'])} (Boulware + Modell {V.signed(adv[6]['Boulware + Modell'])})", f"bei T = 20 {V.signed(adv[20]['Boulware'])} ({V.signed(adv[20]['Boulware + Modell'])})",
        f"bei T = 60 {V.signed(adv[60]['Boulware'])} ({V.signed(adv[60]['Boulware + Modell'])})")


# --- H11 --------------------------------------------------------------------------------------------------------------------------------------
def test_h11_the_learner(data):
    s = {E: R.learner_summary(data, E) for E in C.LEARN_EPISODES}
    b = R.learner_baselines(data)
    tr = R.learner_transfer(data)
    row = row_of("H11")
    has(row, f"{de(s[5000]['mean'], 3)} / {de(s[20000]['mean'], 3)} / {de(s[50000]['mean'], 3)} nach 5.000 / 20.000 / 50.000 Episoden (Std {de(s[5000]['sd'], 3)} / {de(s[20000]['sd'], 3)} / {de(s[50000]['sd'], 3)}) gegen Boulware + Modell {de(b[ST.BOULWARE_MODELL]['mean'], 3)}")
    assert all(r["ahead"] == 4 for r in tr if r["label"] != "s = 1") and [r["ahead"] for r in tr if r["label"] == "s = 1"] == [3] and len(tr) == 6
    has(row, "Übertragung auf sechs Bedingungen: 4 von 4 Trainings-Seeds vorn, bei s = 1 nur 3 von 4")
    hart = s[20000]["per_opp"][ST.HARTNAECKIG][0]
    assert de(hart, 3) == "0,441" and de(b[ST.BOULWARE_MODELL]["per_opp"][ST.HARTNAECKIG], 3) == "0,219"
    behind = [o for o in V.P6 if s[20000]["per_opp"][o][0] < b[ST.BOULWARE_MODELL]["per_opp"][o]]
    assert behind == [ST.LINEAR, ST.CONCEDER]
    has(row, "stark gegen Hartnäckig (0,441 gegen 0,219), hinter Boulware + Modell gegen 2 von 6 Gegnern (Linear, Conceder)")
    assert all(s[E]["mean"] > b[ST.BOULWARE_MODELL]["mean"] for E in C.LEARN_EPISODES) and s[50000]["mean"] < s[20000]["mean"]


# --- H12 --------------------------------------------------------------------------------------------------------------------------------------
def test_h12_counts_from_the_property_tests():
    prop = (ROOT / "tests" / "test_eigenschaften_verhandlungsagenten.py").read_text(encoding="utf-8")
    assert "assert tot == 8 * 36 and mism == 0" in prop and "for k in range(6)" in prop and "range(150)" in prop and "3.5 * max(se" in prop
    assert 8 * 36 == 288 and 6 * 36 == 216
    orakel = (ROOT / "tests" / "test_oracle_verhandlungsagenten.py").read_text(encoding="utf-8")
    assert "assert total == 60 * 6 and bad == 0" in orakel and 60 * 6 == 360 and "assert total == 300 and bad == 0" in orakel
    has(row_of("H12"), "288 Aushandlungen", "216 von 216", "in 6 von 6 Strategien")
    has(section("## Methodik", "## Befunde (gemessen"), "(in 100 Zufallsdomänen)", "(300 Fälle)")


def test_oracle_all_100_domains_are_valid():
    checked = 0
    for k in range(100):
        n, m = [(3, 3), (3, 4), (4, 3), (2, 5)][k % 4]
        rng = np.random.default_rng([11, k])
        d = D.make_domain(D.seed_of(f"orakel|{k}"), n=n, m=m, s=float(rng.random()), g=float(rng.random()), r=0.2 * float(rng.random()))
        checked += D.dom_info(d) is not None
    assert checked == 100


# --- H13 --------------------------------------------------------------------------------------------------------------------------------------
def test_h13_endgame_exploiter(data):
    c7, c9 = R.endgame_cell(data, 20, "7"), R.endgame_cell(data, 20, "9")
    e, m = 6, 5
    row = row_of("H13")
    has(row, f"{de(c7.mean['own'][e], 3)} (Std {de(c7.sd['own'][e], 3)}) gegen {de(c7.mean['own'][m], 3)} für Boulware + Modell, Rang 1 in 4 von 4 Seeds", f"von {de(c7.mean['agree'][m], 3)} auf {de(c7.mean['agree'][e], 3)}",
        f"(Orakel + Endspiel) {de(c9.mean['own'][c9.names.index(ST.ORAKEL_ENDSPIEL)], 3)}")
    assert int((c7.ranks[:, e] == 1).sum()) == 4 and c7.mean["own"][e] > c7.mean["own"][m] + 0.02 and c7.mean["agree"][e] < c7.mean["agree"][m]
    assert c9.top == c9.names.index(ST.ORAKEL_ENDSPIEL)


# --- Domänenstatistik, Gegenprobe, Hilfetexte -------------------------------------------------------------------------------------------------
def test_domain_statistics_rows(dstat):
    rows = dstat
    assert [r["s"] for r in rows] == [0.0, 0.25, 0.5, 0.75, 1.0] and all(r["skipped"] == 0 for r in rows)
    assert rows[-1]["nash_sum"] > rows[0]["nash_sum"] and rows[-1]["max_sum"] > rows[0]["max_sum"] and max(r["max_sum"] - r["nash_sum"] for r in rows) < 0.02
    assert rows[0]["gap"] == pytest.approx(0.0, abs=1e-12)
    sums = [r["nash_sum"] for r in rows]
    assert sums == sorted(sums) and [de(sums[0], 3), de(sums[-1], 3)] == ["1,213", "1,512"]
    assert "1,213 bei s = 0, 1,512 bei s = 1" in T.CONTROLS_TABLE


def test_counter_check_with_other_random_seeds_gives_the_same_top_order():
    """Gegenprobe (§11a Punkt 5): die Hauptzelle mit den Zufalls-Seeds 11 bis 14 aus anderen Bezeichnungen: Boulware + Modell, Boulware, Tit-for-Tat auf Rang 1 bis 3, Conceder Letzter."""
    Ms = []
    for sd in (11, 12, 13, 14):
        doms, _ = D.domain_set(f"gegenprobe|0.5|{sd}", 100, s=0.5)
        Ms.append(TO.pair_means(list(ST.POOL6), doms, 20))
    c = R.cell_stats(np.stack(Ms), R.P6, list(range(6)))
    assert c.ranks[:, 5].tolist() == [1, 1, 1, 1] and c.ranks[:, 1].tolist() == [2, 2, 2, 2] and c.ranks[:, 4].tolist() == [3, 3, 3, 3] and c.ranks[:, 3].tolist() == [6, 6, 6, 6]
    has(README, "mit den Seeds 11 bis 14 statt 1 bis 4", "Boulware + Modell vor Boulware vor Tit-for-Tat auf Rang 1, 2 und 3 in allen vier Seeds, Conceder Letzter")


def test_preset_help_texts_state_only_what_the_presets_do(data):
    h = C.PRESET_HELP
    assert C.PRESETS["Kurze Frist (T = 6)"]["T"] == 6 and "nur 6 Runden" in h["Kurze Frist (T = 6)"] and "führt bei T = 6 Tit-for-Tat" in h["Kurze Frist (T = 6)"]
    assert C.PRESETS["Lange Frist (T = 60)"]["T"] == 60 and "60 Runden" in h["Lange Frist (T = 60)"]
    b = C.BASE
    assert b["n"] == 4 and b["m"] == 4 and b["T"] == 20 and "4 Punkte mit 4 Werten, Frist 20 Runden" in h["Boulware gegen Tit-for-Tat (Voreinstellung)"]
    assert C.PRESETS["Modell wird getäuscht"]["seed"] == 0 and "Zufalls-Seed 0" in h["Modell wird getäuscht"] and C.PRESETS["Endspiel-Ausnutzer"]["seed"] == 0 and "Zufalls-Seed 0" in h["Endspiel-Ausnutzer"]
    assert C.PRESETS["Endspiel-Ausnutzer"]["T"] == 20 and "bei 20 Runden zuletzt" in h["Endspiel-Ausnutzer"] and C.PRESETS["Gegner bietet zuletzt"]["T"] == 20 and "bei 20 Runden zuletzt" in h["Gegner bietet zuletzt"]
    assert NG.make_agent("Linear").spec.beta == 1.0


def test_app_texts_that_state_a_measurement(data):
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    # E5: „Einigungsrate steht nur in der Tabelle: sie unterscheidet sich fast nur zwischen den Kurven, weil Tit-for-Tat seltener einigt“
    e = V.eta_values(data, "alle 48 Agenten", "agree")
    lm = R.level_means(data)
    assert e["Kurve"][0] > 0.99 and lm[("Kurve", "Tit-for-Tat")]["agree"] < min(lm[("Kurve", k)]["agree"] for k in ("Boulware", "Linear", "Conceder")) - 0.2
    assert "sie unterscheidet sich fast nur zwischen den Kurven, weil Tit-for-Tat seltener einigt" in app
    # E2: Hartnäckig/Hartnäckig und Tit-for-Tat/Tit-for-Tat zahlengleich (siehe H5)
    assert "sind zahlengleich" in app
    # CONTROLS_TABLE: E6 misst die genannten Reglerwerte
    labels = [c["label"] for c in R.robust_cells(data)]
    for needle in ("Reservationsnutzen 0", "Reservationsnutzen 0,5", "Gegensatz g = 0", "Gegensatz g = 1", "Fenster 0,02", "Fenster 0,10"):
        assert needle in labels
    assert sum(1 for x in labels if "Punkte ×" in x) == 4 and "E6 prüft vier andere Größen" in T.CONTROLS_TABLE


def test_tests_section_counts_are_in_the_readme():
    sec = section("## Tests\n", "## Dateistruktur")
    assert re.search(r"\*\*\d+ Tests\*\*", sec)
