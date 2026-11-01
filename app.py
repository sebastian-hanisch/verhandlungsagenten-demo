"""Verhandlungsagenten: was trägt gegen unbekannte Gegner? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück der Linie „Regeln setzen, verhandeln, kontrollieren“. Zwei Programme (ein Verlader und ein Spediteur) handeln mehrere Punkte einer Tour aus und kennen die Nutzen des anderen nicht. Die Demo zeigt eine Aushandlung Zug um Zug,
ein Turnier der Strategien, die Rolle der letzten Runde, ein lernendes Gegnermodell, die Zerlegung in Gebot, Annahme und Gegnermodell sowie einen kleinen Q-Lerner.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import vha_constants as C
import vha_evaluation as V
import vha_experiments as XP
import vha_live as LV
import vha_results as R
import vha_strategies as ST
import vha_texts as T
import vha_visualization as Z
from vha_layout import dunkles_design, kopf_css
from vha_presets import (
    STRATEGY_CHOICES,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    settings_from_state,
    sidebar_presets,
    sync_query_params,
)
from vha_presets import SETTING_SPECS

st.set_page_config(page_title="Verhandlungsagenten – Sebastian Hanisch", layout="wide")


# --- Zwischenspeicher (Schlüssel = alle Eingaben inklusive Zufalls-Seed) --------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _data():
    return XP.load()


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return V.analyse(settings)


@st.cache_data(show_spinner=False)
def _model_trace(settings):
    return V.model_trace(V.analyse(settings))


@st.cache_data(show_spinner=False)
def _domain_stats(count):
    return V.domain_stats(count)


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in vha_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🤝 Verhandlungsagenten: was trägt gegen unbekannte Gegner?")
st.markdown(T.INTRO)

with st.expander("So läuft eine Aushandlung in sechs Schritten", expanded=True):
    st.markdown(T.EXPLAINER)

with st.expander("Erklärtabelle: die Strategien"):
    st.markdown(T.STRATEGY_TABLE)
    st.markdown("**Die ersten Gebote jeder Strategie** (Nutzen des eigenen Gebots für den Verlader; Gegner: ein Hartnäckiger, der nie nachgibt; Domäne, Frist und Regel der Seitenleiste)")
    first_bids_slot = st.container()

with st.expander("Erklärtabelle: die Kennzahlen und ihre Bezugsmenge"):
    st.markdown(T.METRICS_TABLE)

with st.expander("Erklärtabelle: was die Regler tun"):
    st.markdown(T.CONTROLS_TABLE)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("📦 Domäne")
    st.slider("Punkte n", *bounds("n_slider"), key="n_slider", step=1, help="Wie viele Punkte verhandelt werden (Preis, Lieferfenster, Zahlungsziel, Haftung, Rückladung in dieser Reihenfolge).")
    st.slider("Werte je Punkt m", *bounds("m_slider"), key="m_slider", step=1, help="Wie viele Werte jeder Punkt hat; es gibt m hoch n Bündel (höchstens 3.125).")
    st.slider("Gewichtsunterschied s", *bounds("s_slider"), key="s_slider", step=0.05, help="0: beide Seiten gewichten die Punkte gleich, 1: unabhängig voneinander. Je verschiedener die Gewichte, desto größer ist in den Zufallsdomänen die Summe der Nutzen im Nash-Bündel (E1).")
    st.slider("Gegensatz der Wertlisten g", *bounds("g_slider"), key="g_slider", step=0.05, help="0: die Wertlisten der beiden Seiten sind unabhängig, 1: genau gegenläufig (was der eine mag, mag der andere nicht).")
    st.slider("Reservationsnutzen r", *bounds("r_slider"), key="r_slider", step=0.05, help="Was jede Seite ohne Einigung bekommt (beide gleich). Nutzen liegen zwischen 0 und 1.")
    st.header("⏱️ Ablauf")
    st.slider("Frist T (Runden)", *bounds("t_slider"), key="t_slider", step=1, help="Zahl der Runden. Der Verlader bietet in den ungeraden Runden, der Spediteur in den geraden; bei geradem T bietet der Spediteur zuletzt.")
    st.slider("Fensterbreite um das Ziel", *bounds("band_slider"), key="band_slider", step=0.01, help="Eine Strategie bietet ein Bündel, dessen Nutzen höchstens so viel über ihrem Ziel liegt.")
    st.selectbox("Regel „letzte Runde“", C.RULE_CHOICES, key="rule_select",
                 help="an: in der letzten Runde nimmt der Antwortende jedes Angebot ab seinem Reservationsnutzen an. aus: er folgt auch dann seiner Annahmeregel. Beide Fassungen sind vertretbar (E3).")
    st.header("🤖 Strategien")
    st.selectbox("Verlader (bietet zuerst)", STRATEGY_CHOICES, key="first_select", help="Die Strategie des Verladers; Erklärung der Strategien oben in der Erklärtabelle.")
    st.selectbox("Spediteur", STRATEGY_CHOICES, key="second_select", help="Die Strategie des Spediteurs.")
    st.number_input("Zufalls-Seed der Domäne", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Domäne generieren", width="stretch", on_click=randomize_seed)

settings = settings_from_state()
sync_query_params({k: st.session_state[k] for k in SETTING_SPECS})

verdicts = []      # Verdikte für den PDF-Export
data = _data()
if data is None:
    st.warning("Die vorgerechneten Messreihen (data/messreihen.json) fehlen oder gehören zu einem anderen Rechencode: `python tools/make_grid.py` ausführen. Die Experimente zeigen dann nur die Live-Teile.")

view = _analysis(settings)
with first_bids_slot:
    table(V.first_bid_rows(settings))

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------
st.markdown("## 🎯 Ergebnis: eine Aushandlung")
r = view.result
if view.info is None:
    st.metric("Ausgang", "keine Einigung möglich", help="Kein Bündel bringt beiden Seiten mindestens den Reservationsnutzen.")
else:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Ausgang", f"Runde {r.t} von {settings.T}" if r.agree else "keine Einigung", help="Runde, in der angenommen wurde. Ohne Einigung bekommen beide den Reservationsnutzen.")
    m2.metric("Nutzen Verlader", V.de(r.u0, 3), help=f"Strategie: {settings.first}.")
    m3.metric("Nutzen Spediteur", V.de(r.u1, 3), help=f"Strategie: {settings.second}.")
    m4.metric("Summe", V.de(view.ev["wel"], 3), help=f"Größte Summe aller rationalen Bündel in dieser Domäne: {V.de(view.info['maxsum'], 3)}.")
    m5.metric("Abstand zur Grenze", V.de(view.ev["dist"], 3), help="Kleinster Abstand des Ergebnisses zu einem Punkt der Pareto-Grenze (Grenze der Bündel, die beiden mindestens den Reservationsnutzen bringen).")
col_a, col_b = st.columns([3, 2])
with col_a:
    st.plotly_chart(Z.build_offers(view), width="stretch", key="offers_chart")
    st.caption("Nutzenebene: jedes graue Feld ist ein Bündel, blau die Pareto-Grenze der individuell rationalen Bündel, violett die Nash-Lösung. Die Zahlen an den Angeboten sind die Runden; "
               "der Verlader (grün) bietet in den ungeraden, der Spediteur (orange) in den geraden Runden. Gestrichelt: Reservationsnutzen.")
with col_b:
    table(V.trace_rows(view))
    st.caption("Bündel = Wertstufe je Punkt (0 bis m − 1; Punkte in der Reihenfolge Preis, Lieferfenster, …). Das Ziel ist der Zielnutzen des Bietenden vor dem Gebot.")
st.markdown("#### ❓ Kernfrage: Einigen sich diese beiden Programme, und wie weit von der Grenze?")
v_main = V.verdict_negotiation(view)
show(v_main)
verdicts.append(v_main)
with st.expander("Zielkurven dieser Aushandlung und die Familie der Zeitkurven"):
    st.plotly_chart(Z.build_targets(view), width="stretch", key="targets_chart")
    st.caption("Linie: Ziel des Bietenden je Runde (Tit-for-Tat und Hartnäckig haben keine Zeitkurve), Kreuze: Nutzen des erhaltenen Angebots für den Antwortenden. "
               "Angenommen wird in der Regel, wenn das Kreuz auf oder über dem nächsten eigenen Gebot liegt.")
    st.plotly_chart(Z.build_curve_family(settings.r), width="stretch", key="curve_family_chart")
    st.caption("Die Zeitkurven U(t) = 1 − (1 − r) · τ^(1/β) über dem Zeitfortschritt τ (Formelfamilie laut Sekundärliteratur, nicht im Volltext geprüft).")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1 Domäne und Orakel", "2 Turnier", "3 Wer zuletzt bietet", "4 Gegnermodell", "5 Bausteine zerlegen", "6 Lernender 🧪"])

# --- Tab 1 ---------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("🔬 Experiment E1: Domäne und Orakel")
    st.caption("**Erwartung:** Pareto-Grenze und Nash-Lösung lassen sich durch Aufzählen aller Bündel finden und stimmen mit einem zweiten Rechenweg überein; je verschiedener die Seiten gewichten, desto mehr ist zu holen. "
               "Die Strategien kennen weder Grenze noch Nash-Lösung, sie sind nur der Maßstab.")
    table(V.summary_rows(view))
    st.caption(f"Aufgezählt werden alle {V.thousands(view.dom.N)} Bündel dieser Domäne. Ein zweiter Rechenweg (Schleifen über alle Paare, Maximierer gewichteter Summen, Handbeispiel) steht in den Tests.")
    if st.button("📊 Domänenstatistik rechnen", key="dstat_button"):
        st.session_state["dstat"] = _domain_stats(400)
    if st.session_state.get("dstat"):
        table(V.domain_stats_rows(st.session_state["dstat"]))
        st.caption("400 Zufallsdomänen je Zeile (4 Punkte × 4 Werte, g = 0,5, r = 0,2); Gewichtsabstand = ½‖w₀ − w₁‖₁.")
        v_dom = V.verdict_domain(st.session_state["dstat"], 400)
        show(v_dom)
        verdicts.append(v_dom)

# --- Tab 2 ---------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("🔬 Experiment E2: Turnier der Strategien")
    st.caption("**Erwartung:** Wer früh nachgibt (Conceder), verliert; zwei Hartnäckige einigen sich kaum; die Rangfolge hängt an Frist und Gewichtsunterschied. "
               f"Sechs Strategien, jede gegen jede in beiden Positionen, 4 Zufalls-Seeds zu je {C.TURNIER_DOMAINS} Domänen (4 Punkte × 4 Werte, g = 0,5, r = 0,2, Regel „letzte Runde“ an); Wert = Mittel über die fünf anderen Strategien.")
    if data:
        t_pick = st.selectbox("Frist T", C.TURNIER_TS, index=1, key="e2_t")
        s_pick = st.selectbox("Gewichtsunterschied s", C.TURNIER_SS, index=1, key="e2_s")
        cell = R.turnier_cell(data, t_pick, s_pick)
        table(V.turnier_table(cell))
        st.caption(f"Zelle {V.stage_label(t_pick, s_pick)}. Std = Standardabweichung über die vier Zufalls-Seeds; Rang je Seed in der Reihenfolge der Seeds 1 bis 4.")
        st.plotly_chart(Z.build_bars([V.short(n) for n in cell.names], cell.mean["own"], cell.sd["own"], "eigener Nutzen", highlight=V.short(cell.names[cell.top]), xrange=[0, 0.8]), width="stretch", key="e2_bars")
        st.markdown("**Sieger je Zelle** (in Klammern: in wie vielen der vier Seeds er auch einzeln auf Rang 1 liegt)")
        table(V.winners_table(data))
        v2 = V.verdict_turnier(data)
        show(v2)
        verdicts.append(v2)
        metric = st.selectbox("Paarmatrix in dieser Zelle", ("Nutzen der zuerst bietenden Strategie", "Nutzen der zweiten Strategie", "Einigungsrate"), key="e2_pair")
        key_m = {"Nutzen der zuerst bietenden Strategie": "u0", "Nutzen der zweiten Strategie": "u1", "Einigungsrate": "agree"}[metric]
        st.plotly_chart(Z.build_heat(R.pair_matrix(cell, key_m), [V.short(n) for n in cell.names], [V.short(n) for n in cell.names], "Wert", zmin=0, zmax=1, fmt=".2f"), width="stretch", key="e2_heat")
        st.caption("Zeile = die Strategie, die in Runde 1 bietet; Spalte = die zweite. Die Diagonale ist das Selbstspiel. Hartnäckig gegen Hartnäckig und Tit-for-Tat gegen Tit-for-Tat sind zahlengleich: "
                   "Tit-for-Tat gibt nur nach, wenn der andere nachgibt, und ist im Selbstspiel hartnäckig.")
    st.markdown("**Ein Pool-Turnier live** (Domäne, Frist, Regel und Fensterbreite der Seitenleiste)")
    c_pool, c_count = st.columns(2)
    live_pool = c_pool.selectbox("Pool", ("6", "7"), format_func=lambda x: "6 Strategien" if x == "6" else "6 Strategien und Endspiel-Ausnutzer", key="live_pool")
    live_count = c_count.slider("Domänen je Paar", C.LIVE_DOMAINS_MIN, C.LIVE_DOMAINS_MAX, C.LIVE_DOMAINS_DEFAULT, step=10, key="live_count")
    st.caption("Dauer je nach Domäne und Frist von unter einer Sekunde bis rund einer halben Minute (Einzelmessungen: README, Abschnitt „Rechenzeit“).")
    if st.button("▶️ Pool-Turnier rechnen", key="live_tournament_button"):
        with st.spinner("Turnier läuft …"):
            st.session_state["live_tournament"] = (settings, live_pool, live_count, LV.live_tournament(settings, live_pool, live_count))
    lt = st.session_state.get("live_tournament")
    if lt and lt[0] == settings and lt[1] == live_pool and lt[2] == live_count:
        res = lt[3]
        order = sorted(range(len(res["names"])), key=lambda i: -res["scores"]["own"][i])
        table([{"Strategie": V.short(res["names"][i]), "Eigener Nutzen": V.de(res["scores"]["own"][i], 3), "Summe": V.de(res["scores"]["wel"][i], 3), "Abstand zur Grenze": V.de(res["scores"]["dist"][i], 3),
                "Einigungsrate": V.de(res["scores"]["agree"][i], 3)} for i in order])
        st.caption(f"{V.thousands(res['negotiations'])} Aushandlungen ({len(res['names'])} × {len(res['names'])} Paare × {res['count']} Domänen), eine Stichprobe mit einem Zufalls-Seed: Mittel und Std über vier Seeds stehen im Raster oben.")
    elif lt:
        st.caption("Die Einstellungen haben sich seit dem letzten Lauf geändert: erneut rechnen.")
    if data:
        st.markdown("---")
        st.subheader("🔬 Experiment E6: Hält die Rangfolge, wenn man einen Regler ändert?")
        st.caption("**Erwartung:** Die Rangfolge hält. Basis: T = 20, s = 0,5, 4 Punkte × 4 Werte, r = 0,2, g = 0,5, Fenster 0,05; jede Zeile ändert genau einen Regler (4 Zufalls-Seeds zu je 50 Domänen).")
        table(V.robust_rows(data))
        v6 = V.verdict_robust(data)
        show(v6)
        verdicts.append(v6)

# --- Tab 3 ---------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Experiment E3: Wer zuletzt bietet")
    st.caption("**Erwartung:** Wer hartnäckig bleibt, wird belohnt, weil der Gegner nachgibt. Hartnäckig gegen die anderen, getrennt nach Position und Parität der Frist (bei geradem T bietet die zweite Position zuletzt, bei ungeradem die erste), "
               "4 Zufalls-Seeds zu je 100 Domänen, s = 0,5.")
    if data:
        rule_pick = st.radio("Regel „letzte Runde“", ("an", "aus"), horizontal=True, key="e3_rule")
        table(V.position_table(data, rule_pick == "an"))
        st.caption("Gegner: Boulware, Tit-for-Tat (relativ) und Hartnäckig; gegen Linear und Conceder ergeben sich in allen Zellen dieselben Nutzen und Einigungsraten wie gegen Boulware, nur die mittlere Einigungsrunde unterscheidet sich. Std = Standardabweichung der Nutzen der Hartnäckigen über die Seeds.")
        v3 = V.verdict_position(data)
        show(v3)
        verdicts.append(v3)
        st.markdown("**Die Regel „letzte Runde“ als Auswahl (A oder B) mit Wirkungstabelle** (Pool-Turnier T = 20, s = 0,5, dieselben Domänen)")
        st.markdown("- **A, an:** in Runde T nimmt der Antwortende jedes Angebot ab seinem Reservationsnutzen an. Vorteil: ein Ablauf endet nicht mit einem Angebot, das beide annehmen würden. Nachteil: das letzte Gebot wird zum Alles-oder-nichts für den Antwortenden.\n"
                    "- **B, aus:** der Antwortende folgt auch in Runde T seiner Annahmeregel. Vorteil: die letzte Runde ist eine Runde wie die anderen. Nachteil: Einigungen scheitern, die beide wollten (siehe Einigungsrate).")
        table(V.rule_rows(data))
        v_rule = V.verdict_rule(data)
        show(v_rule)
        verdicts.append(v_rule)
        st.markdown("**Endspiel-Ausnutzer:** das letzte Gebot ist das für mich beste Bündel, das der Gegner laut Modell noch annimmt (nur sinnvoll, wenn man zuletzt bietet und die Regel „letzte Runde“ an ist).")
        e_t = st.selectbox("Frist T", C.ENDSPIEL_TS, key="e3_endgame_t")
        e_pool = st.selectbox("Pool", ("6", "7", "9"), index=1, key="e3_endgame_pool",
                              format_func=lambda x: {"6": "6 Standardstrategien", "7": "7: dazu der Endspiel-Ausnutzer", "9": "9: dazu zwei Orakel-Agenten (kennen den wahren Nutzen des Gegners)"}[x])
        table(V.endgame_rows(R.endgame_cell(data, e_t, e_pool)))
        st.caption("4 Zufalls-Seeds zu je 50 Domänen, s = 0,5, Regel „letzte Runde“ an. Der Endspiel-Ausnutzer schätzt den Nutzen des Gegners mit dem Häufigkeitsmodell (grobe Annäherung: die Schätzung liegt nicht auf der Nutzenskala).")
        v_end = V.verdict_endgame(data)
        show(v_end)
        verdicts.append(v_end)

# --- Tab 4 ---------------------------------------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Experiment E4: Was lernt ein Gegnermodell?")
    st.caption("**Erwartung:** Das Häufigkeitsmodell lernt die Gewichte des Gegners nach wenigen Angeboten und hebt vor allem den eigenen Nutzen. "
               "Der Beobachter spielt Linear und nimmt nie an (T = 40), der Gegner bietet; Güte nach k Angeboten, Mittel über 4 Zufalls-Seeds zu je 100 Domänen.")
    st.markdown("**An dieser Aushandlung:** was ein Häufigkeitsmodell jeder Seite nach den Angeboten der anderen geschätzt hätte (auch wenn die Strategie keins nutzt)")
    mt = _model_trace(settings)
    if mt:
        st.plotly_chart(Z.build_lines({who: ([r_["k"] for r_ in mt if r_["Beobachter"] == who], [r_["rho"] for r_ in mt if r_["Beobachter"] == who], None) for who in ("Verlader", "Spediteur")},
                                      "gesehene Angebote k", "Rangkorrelation û gegen wahren Nutzen", yrange=[-1.05, 1.05], hline=(0.0, "kein Zusammenhang"), colors={"Verlader": C.COLOR_FIRST, "Spediteur": C.COLOR_SECOND}), width="stretch", key="model_live_chart")
    v_ml = V.verdict_model_live(view, mt)
    show(v_ml)
    if data:
        s_model = st.selectbox("Gewichtsunterschied s", C.MODEL_SS, key="e4_s")
        series_gap, series_rho = {}, {}
        for opp in R.EFFECT_OPPS:
            m = R.model_curve(data, opp, s_model)
            series_gap[V.short(opp)] = (m["k"], m["gap"], None)
            series_rho[V.short(opp)] = (m["k"], m["rho"], None)
        col_g, col_r = st.columns(2)
        with col_g:
            st.plotly_chart(Z.build_lines(series_gap, "gesehene Angebote k", "Gewichtsabstand ½‖ŵ − w‖₁", yrange=[0.1, 0.4], categorical=True, hline=(R.model_curve(data, ST.LINEAR, s_model)["guess"], "gleiche Gewichte raten")), width="stretch", key="model_gap_chart")
            st.caption(f"Gewichtsabstand: kleiner ist besser. Gleiche Gewichte raten gibt {V.de(R.model_curve(data, ST.LINEAR, s_model)['guess'], 3)}.")
        with col_r:
            st.plotly_chart(Z.build_lines(series_rho, "gesehene Angebote k", "Rangkorrelation û gegen wahren Nutzen", yrange=[-1.0, 1.0], categorical=True, hline=(0.0, "kein Zusammenhang")), width="stretch", key="model_rho_chart")
            st.caption("Rangkorrelation: 1 heißt, das Modell ordnet alle Bündel wie der Gegner. Gegen Conceder ist sie bei 12 und bei 20 Angeboten negativ.")
        table(V.model_rows(data, s_model))
        st.markdown("**Wirkung auf das Ergebnis** (Programm gegen die fünf Basisgegner, beide Positionen, T = 20, s = 0,5; Orakel kennt den wahren Nutzen des Gegners)")
        table(V.effect_rows(data))
        v4 = V.verdict_model(data)
        show(v4)
        verdicts.append(v4)

# --- Tab 5 ---------------------------------------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("🔬 Experiment E5: Bausteine zerlegen – Gebot, Annahme, Gegnermodell")
    st.caption("**Erwartung (aus einem Abstract, nicht geprüft):** das Gebot wiegt mehr als die Annahme, die Annahme mehr als das Gegnermodell. "
               "48 Agenten = 4 Kurven × 4 Annahmeregeln × 3 Gegnermodelle (keins, Häufigkeit, Orakel), jeder gegen die sechs Pool-Strategien in beiden Positionen (T = 20, s = 0,5), 4 Zufalls-Seeds zu je 50 Domänen. "
               "Varianzanteil η² des Faktors an den Agentenmitteln; „Rest“ sind Wechselwirkungen. **Der Anteil hängt an der Kennzahl und an den gewählten Stufen.** "
               "(In der Literatur heißt diese Zerlegung laut Verzeichnis BOA: Bidding strategy, Opponent model, Acceptance strategy; nicht zu verwechseln mit BOA* = Bi-Objective A* in `multicriteria-demo`.)")
    if data:
        stage = st.selectbox("Stufen", list(R.STAGE_SETS), key="e5_stage")
        table(V.eta_rows(data, stage))
        arrs, combos = R.decomp_arrays(data), R.decomp_combos()
        parts = {f: [] for f in (*R.FACTORS, "Rest")}
        labels = []
        for m_, lab in R.DECOMP_METRICS.items():
            if m_ == "agree":
                continue
            e = R.eta_squared(arrs[m_], combos, R.STAGE_SETS[stage])
            labels.append(lab)
            for f in parts:
                parts[f].append(e[f][0])
        st.plotly_chart(Z.build_stacked(labels, parts, "Anteil an der Varianz (η²)"), width="stretch", key="e5_eta_chart")
        st.caption("Einigungsrate steht nur in der Tabelle: sie unterscheidet sich fast nur zwischen den Kurven, weil Tit-for-Tat seltener einigt.")
        v5 = V.verdict_decomp(data)
        show(v5)
        verdicts.append(v5)
        st.markdown("**Annahmeregeln im Einzelnen** (Kurve Linear, kein Modell, gegen den Pool)")
        table(V.accept_rows(data))
        best, worst = R.best_worst_agents(data)
        st.markdown("**Die fünf besten und die fünf schlechtesten der 48 Agenten** (eigener Nutzen)")
        table([{"Kurve": b[0], "Annahme": b[1], "Modell": b[2], "Eigener Nutzen (Std)": V.pm(b[3], b[4])} for b in best] + [{"Kurve": "…", "Annahme": "…", "Modell": "…", "Eigener Nutzen (Std)": "…"}]
              + [{"Kurve": w[0], "Annahme": w[1], "Modell": w[2], "Eigener Nutzen (Std)": V.pm(w[3], w[4])} for w in worst])

# --- Tab 6 ---------------------------------------------------------------------------------------------------------------------------------
with tab6:
    st.subheader("🧪 Experiment E7: Ein Lernender (gelb: nur gegen diesen Pool und Generator)")
    st.caption("**Erwartung:** Ein Q-Lerner schlägt die festen Strategien stabil. Zustand: 90 (Zeit, Nutzen des letzten Angebots, Trend), Aktion: Zielnutzen aus 9 Stufen, Belohnung: Nutzen am Ende; Training gegen die fünf Basisgegner in "
               "frischen Domänen (T = 20, s = 0,5), Test an neuen Domänen gegen alle sechs Pool-Strategien. **Nur Bandbreiten, keine Aussage über andere Gegner.**")
    if data:
        table(V.learner_rows(data))
        st.caption("Jedes Training und jeder Test hat vier Seeds (Std über die Trainings-Seeds); die festen Strategien auf denselben Test-Domänen. Die Trainingsläufe sind getrennt, jeder mit voller ε-Abnahme.")
        table(V.learner_opponent_rows(data))
        st.markdown("**Übertragung** auf andere Bedingungen (Training mit 20.000 Episoden bei T = 20, s = 0,5)")
        table(V.transfer_rows(data))
        st.markdown("**Gegner im Training ausgelassen** (20.000 Episoden)")
        table(V.omitted_rows(data))
        v7 = V.verdict_learner(data)
        show(v7)
        verdicts.append(v7)
    st.markdown("**Live trainieren** (fünf Basisgegner, Standarddomäne, Frist 20; Test in der Domäne und mit Frist, Regel und Fenster der Seitenleiste; Zufalls-Seed der Seitenleiste)")
    if st.button("▶️ Lernenden trainieren", key="live_learner_button"):
        with st.spinner(f"{V.thousands(C.LIVE_EPISODES)} Episoden …"):
            st.session_state["live_learner"] = (settings, LV.live_learner(settings))
    ll = st.session_state.get("live_learner")
    if ll and ll[0] == settings:
        res = ll[1]
        table([{"Gegner": V.short(o), "Lernender": V.de(res["test"][o][0], 3), "Boulware + Modell": V.de(res["base"][o], 3)} for o in ST.POOL6]
              + [{"Gegner": "Mittel über die sechs", "Lernender": V.de(res["mean"], 3), "Boulware + Modell": V.de(res["base_mean"], 3)}])
        st.caption(f"{V.thousands(res['episodes'])} Episoden, {res['count']} Test-Domänen, ein Trainings-Seed: ein einzelner Lauf, keine Aussage; die Reihen oben haben vier.")
    elif ll:
        st.caption("Die Einstellungen haben sich seit dem letzten Lauf geändert: erneut trainieren.")

# --- Verwandte Demos, Grenzen, Formeln -----------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Verwandte Demos")
st.markdown(
    "- [`verhandlung-demo`](https://github.com/sebastian-hanisch/verhandlung-demo): das Gleichgewicht eines Ablaufs mit **einem** Anteil und bekannten Nutzen; hier spielen Programme ohne Kenntnis der Nutzen des Gegners.\n"
    "- [`kuchen-vergroessern-demo`](https://github.com/sebastian-hanisch/kuchen-vergroessern-demo): woher der Gewinn bei mehreren Punkten kommt; hier wird gemessen, welche Strategie ihn hebt.\n"
    "- `axelrod-turnier-demo`: 16 Strategien im wiederholten Gefangenendilemma, kein Verhandlungsablauf mit Frist.\n"
    "- `tit-for-tat-demo`: Tit-for-Tat als Automat im Dilemma; hier ist es ein relatives Nachgeben im Verhandlungsverlauf.\n"
    "- `noregret-demo`: Lernen ohne Reue in einem anderen Spiel."
)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS_MARKDOWN)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.MATH_MARKDOWN)

st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from vha_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(settings, view, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="verhandlungsagenten.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
