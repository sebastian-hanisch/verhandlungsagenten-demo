"""Konstanten: Regler-Grenzen, feste Annahmen, Messparameter, Farben und Voreinstellungen.

Rechenaufwand am Slider-Maximum (DEMO-PLAYBOOK §1c): eine Aushandlung rechnet je Runde über alle `m ** n` Bündel (höchstens 5 ** 5 = 3.125) einige Vektoroperationen, bei höchstens
T = 100 Runden also O(T · m^n · n); gemessen im Bau: siehe README („Rechenzeit“). Die Messreihen (Turnier-Raster, Endspiel, Position, Gegnermodell, Zerlegung, Robustheit, Lernender)
sind vorgerechnet (`tools/make_grid.py`, Dateien in `data/`); live laufen eine Aushandlung, ein Pool-Turnier mit wenigen Domänen und ein kurzes Training des Lernenden.
"""

# --- Regler -------------------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 3, 5, 4                      # Punkte der Tour (Preis, Lieferfenster, Zahlungsziel, Haftung, Rückladung)
M_MIN, M_MAX, DEFAULT_M = 3, 5, 4                      # Werte je Punkt
S_MIN, S_MAX, DEFAULT_S = 0.0, 1.0, 0.5                # Gewichtsunterschied
G_MIN, G_MAX, DEFAULT_G = 0.0, 1.0, 0.5                # Gegensatz der Wertlisten
R_MIN, R_MAX, DEFAULT_R = 0.0, 0.5, 0.2                # Reservationsnutzen (beide Seiten gleich)
T_MIN, T_MAX, DEFAULT_T = 2, 100, 20                   # Frist in Runden (T = 100 mit 5 Punkten × 5 Werten: 3.125 Bündel)
BAND_MIN, BAND_MAX, DEFAULT_BAND = 0.02, 0.10, 0.05    # Fensterbreite um das Ziel (Isolinie)
SEED_MAX, DEFAULT_SEED = 999_999, 35
RULE_CHOICES = ("an", "aus")                           # Regel „letzte Runde“ (Antwortender nimmt in Runde T alles ab Reservation an)
DEFAULT_RULE = "an"
POINT_NAMES = ("Preis", "Lieferfenster", "Zahlungsziel", "Haftung", "Rückladung")
LIVE_DOMAINS_MIN, LIVE_DOMAINS_MAX, LIVE_DOMAINS_DEFAULT = 10, 50, 30     # Domänen je Paar im Live-Turnier (Reglergrenzen im Turnier-Reiter)
LIVE_EPISODES = 5000                                   # Episoden des Live-Trainings des Lernenden

# --- Messreihen (Teil des Erzeuger-Hashes) -----------------------------------------------------------------------------------------------
SEEDS = (1, 2, 3, 4)                                   # Zufalls-Seeds der Messreihen: je Zelle vier unabhängige Blöcke
TURNIER_TS = (6, 20, 60)
TURNIER_SS = (0.0, 0.5, 1.0)
TURNIER_DOMAINS = 100                                  # Domänen je Zufalls-Seed und Gewichtsunterschied (dieselben für alle Fristen)
ENDSPIEL_TS = (20, 21)
ENDSPIEL_DOMAINS = 50
POSITION_TS = (6, 7, 20, 21)
POSITION_DOMAINS = 100
MODEL_KS = (1, 2, 3, 5, 8, 12, 20)                     # nach k Angeboten des Gegners
MODEL_SS = (0.5, 1.0)
MODEL_DOMAINS = 100
MODEL_HORIZON = 40                                     # lange Angebotsfolgen: der Beobachter nimmt nie an
EFFECT_DOMAINS = 100
EFFECT_HORIZON = 20
DECOMP_DOMAINS = 50
DECOMP_HORIZON = 20
ROBUST_DOMAINS = 50
ROBUST_HORIZON = 20
LEARN_EPISODES = (5000, 20000, 50000)
LEARN_TEST_DOMAINS = 100
LEARN_HORIZON = 20
LEARN_REF_EPISODES = 20000                             # Übertragung und Auslassen gelten für diese Trainingslänge
THETA = 0.7                                            # Schwelle der Annahmeregel „Schwelle“ in der Zerlegung

# --- ab hier nicht mehr im Erzeuger-Hash der Messreihen (Farben, Voreinstellungen) ------------------------------------------------------
COLOR_FIRST = "#2F6B65"
COLOR_SECOND = "#C26B3A"
COLOR_FRONT = "#4c78a8"
COLOR_OTHER = "rgba(150,150,150,0.45)"
COLOR_NASH = "#8c4f9e"

BASE = {"n": DEFAULT_N, "m": DEFAULT_M, "s": DEFAULT_S, "g": DEFAULT_G, "r": DEFAULT_R, "T": DEFAULT_T, "band": DEFAULT_BAND, "rule": DEFAULT_RULE,
        "first": "Boulware", "second": "Tit-for-Tat (relativ)", "seed": DEFAULT_SEED}

PRESETS = {
    "Boulware gegen Tit-for-Tat (Voreinstellung)": dict(BASE),
    "Zwei Sturköpfe": {**BASE, "first": "Hartnäckig", "second": "Hartnäckig"},
    "Kurze Frist (T = 6)": {**BASE, "T": 6},
    "Lange Frist (T = 60)": {**BASE, "T": 60, "second": "Boulware + Modell"},
    "Gegner bietet zuletzt": {**BASE, "first": "Hartnäckig", "second": "Linear"},
    "Modell wird getäuscht": {**BASE, "first": "Boulware + Modell", "second": "Conceder", "seed": 0},
    "Gleiche Gewichte": {**BASE, "s": 0.0},
    "Endspiel-Ausnutzer": {**BASE, "first": "Boulware", "second": "Boulware + Modell + Endspiel", "seed": 0},
}
PRESET_HELP = {
    "Boulware gegen Tit-for-Tat (Voreinstellung)": "Der Verlader hält sein Ziel lange hoch (Boulware), der Spediteur gibt nur nach, wenn der andere nachgibt (Tit-for-Tat, relativ). 4 Punkte mit 4 Werten, Frist 20 Runden.",
    "Zwei Sturköpfe": "Beide Seiten bieten immer ihr eigenes Maximum und nehmen nur ein Angebot an, das so gut ist wie ihr eigenes. Eine Einigung gibt es hier nur durch die Regel „letzte Runde“ (mit ihr aus: keine Einigung).",
    "Kurze Frist (T = 6)": "Dieselben Strategien, aber nur 6 Runden. In der Turnierwertung (E2) führt bei T = 6 Tit-for-Tat.",
    "Lange Frist (T = 60)": "60 Runden; der Spediteur ist Boulware mit Häufigkeitsmodell: er schätzt aus den Angeboten, welche Bündel dem Verlader gefallen, und bietet unter gleich guten Bündeln das, das dem Verlader laut Schätzung am liebsten ist.",
    "Gegner bietet zuletzt": "Der Verlader ist hartnäckig und bietet zuerst, der Spediteur (linear) bietet bei 20 Runden zuletzt und gibt in der letzten Runde bis zum Reservationsnutzen nach; ohne die Regel „letzte Runde“ gäbe es in dieser Domäne keine Einigung (E3).",
    "Modell wird getäuscht": "Der Verlader nutzt das Häufigkeitsmodell gegen einen Conceder. Der Conceder bietet bald Bündel, die dem Verlader gefallen: das Modell lernt dessen Vorlieben statt der des Gegners (Reiter 4, Zufalls-Seed 0: die Rangkorrelation des Modells des Verladers wird negativ).",
    "Gleiche Gewichte": "Beide Seiten gewichten die Punkte gleich (Gewichtsunterschied 0); die Wertlisten unterscheiden sich weiter. Was das für die Summe der Nutzen heißt, zeigt E1.",
    "Endspiel-Ausnutzer": "Der Spediteur bietet bei 20 Runden zuletzt und wählt in der letzten Runde das für ihn beste Bündel, das der Verlader laut Modell noch annimmt (Regel „letzte Runde“ an). In dieser Domäne (Zufalls-Seed 0) bekommt er so mehr als mit Boulware + Modell ohne Endspiel; der Mittelwert über viele Domänen steht in E3.",
}
