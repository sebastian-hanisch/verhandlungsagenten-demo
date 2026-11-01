"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Kennzahlen der Aushandlung, Gebotsfolge und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import vha_evaluation as V
from vha_scenario import Settings

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "ε": "eps", "β": "beta", "α": "alpha", "θ": "theta", "λ": "lambda", "σ": "sigma", "π": "pi", "δ": "delta", "η": "eta", "Δ": "Delta", "γ": "gamma", "ρ": "rho",
            "≈": "~", "½": "1/2", "…": "...", "‚": ",", "‘": "'", "’": "'", "”": '"', "µ": "mu", "∈": "in", "τ": "tau", "û": "u", "ŵ": "w", "×": "x", "⁎": "*", "✓": "ok"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1 (Umlaute gehören dazu): Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(st: Settings, view: V.LiveView, verdicts: list[tuple[str, str]]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Verhandlungsagenten: was trägt gegen unbekannte Gegner?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Domäne: {st.n} Punkte mit je {st.m} Werten, Gewichtsunterschied s = {V.de(st.s, 2)}, Gegensatz g = {V.de(st.g, 2)}, Reservationsnutzen {V.de(st.r, 2)}, Zufalls-Seed {st.seed}. "
                                f"Frist {st.T} Runden, Regel „letzte Runde“ {st.rule}, Fenster {V.de(st.band, 2)}. Verlader: {st.first} (bietet zuerst). Spediteur: {st.second}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Kennzahlen der Aushandlung", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for row in V.summary_rows(view):
        pdf.multi_cell(0, 5, _latin(f"{row['Kennzahl']}: {row['Wert']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Gebotsfolge", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for row in V.trace_rows(view)[:60]:
        pdf.multi_cell(0, 4.5, _latin(f"Runde {row['Runde']}: {row['Bietet']} bietet {row['Bündel']} (Verlader {row['Nutzen Verlader']}, Spediteur {row['Nutzen Spediteur']}), {row['Antwort']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Strategien, Nutzen und Fristen sind Modellannahmen, keine Aussagen über echte Verhandler oder Sprachmodelle. Pareto-Grenze und Nash-Lösung sind durch Aufzählen aller Bündel "
                                "gerechnet; die Messreihen sind Simulationen mit festen Zufalls-Seeds."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
