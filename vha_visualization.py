"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import vha_constants as C
import vha_strategies as ST


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend_below=True, legend_y=-0.22):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=legend_y) if legend_below else dict())
    return lock_axes(fig)


def build_offers(view):
    """Nutzenebene (Verlader waagerecht, Spediteur senkrecht): alle Bündel grau, Pareto-Grenze der individuell rationalen Bündel, Nash-Lösung, Reservationsnutzen,
    die Angebote beider Seiten in der Reihenfolge der Runden (Zahl = Runde) und die Einigung."""
    dom, info = view.dom, view.info
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=dom.u[0], y=dom.u[1], mode="markers", marker=dict(size=4, color=C.COLOR_OTHER), name="alle Bündel", hoverinfo="skip"))
    if info is not None:
        F = info["F"][np.argsort(info["F"][:, 0])]
        fig.add_trace(go.Scatter(x=F[:, 0], y=F[:, 1], mode="lines+markers", line=dict(color=C.COLOR_FRONT, width=2, shape="hv"), marker=dict(size=6), name="Pareto-Grenze (rationale Bündel)",
                                 hovertemplate="Verlader %{x:.3f} | Spediteur %{y:.3f}<extra></extra>"))
        nb = info["nash"]
        fig.add_trace(go.Scatter(x=[dom.u[0][nb]], y=[dom.u[1][nb]], mode="markers", marker=dict(size=14, color=C.COLOR_NASH, symbol="diamond", line=dict(width=1, color="white")), name="Nash-Lösung",
                                 hovertemplate="Nash: %{x:.3f} | %{y:.3f}<extra></extra>"))
    fig.add_vline(x=float(dom.r[0]), line=dict(color="gray", dash="dot", width=1), annotation_text="Reservationsnutzen Verlader", annotation_position="top right")
    fig.add_hline(y=float(dom.r[1]), line=dict(color="gray", dash="dot", width=1), annotation_text="Reservationsnutzen Spediteur", annotation_position="top right")
    for p, color, name in ((0, C.COLOR_FIRST, "Angebote des Verladers"), (1, C.COLOR_SECOND, "Angebote des Spediteurs")):
        rows = [(t, o) for t, q, o, _, _ in view.trace if q == p]
        if rows:
            fig.add_trace(go.Scatter(x=[float(dom.u[0][o]) for _, o in rows], y=[float(dom.u[1][o]) for _, o in rows], mode="markers+text", text=[str(t) for t, _ in rows], textposition="top center",
                                     marker=dict(size=9, color=color, line=dict(width=1, color="white")), name=name, hovertemplate="Runde %{text}: %{x:.3f} | %{y:.3f}<extra></extra>"))
    if view.result.agree:
        fig.add_trace(go.Scatter(x=[view.result.u0], y=[view.result.u1], mode="markers", marker=dict(size=17, color="gold", symbol="star", line=dict(width=1, color="black")), name="Einigung",
                                 hovertemplate="Einigung: %{x:.3f} | %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Nutzen des Verladers", range=[-0.03, 1.05])
    fig.update_yaxes(title_text="Nutzen des Spediteurs", range=[-0.03, 1.05])
    return _base(fig, 480, legend_y=-0.3)


def build_targets(view):
    """Ziel des Bietenden je Runde (Linie) und Nutzen des erhaltenen Angebots für die Antwortenden (Punkte); gestrichelt der Reservationsnutzen."""
    dom = view.dom
    fig = go.Figure()
    for p, color, name in ((0, C.COLOR_FIRST, "Verlader"), (1, C.COLOR_SECOND, "Spediteur")):
        rows = [(t, goal) for t, q, _, _, goal in view.trace if q == p and goal == goal]
        if rows:
            fig.add_trace(go.Scatter(x=[t for t, _ in rows], y=[g for _, g in rows], mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=6), name=f"Ziel {name}",
                                     hovertemplate=f"Ziel {name}, Runde %{{x}}: %{{y:.3f}}<extra></extra>"))
        got = [(t, float(dom.u[p][o])) for t, q, o, _, _ in view.trace if q != p]
        if got:
            fig.add_trace(go.Scatter(x=[t for t, _ in got], y=[u for _, u in got], mode="markers", marker=dict(size=8, color=color, symbol="x"), name=f"erhaltenes Angebot {name}",
                                     hovertemplate=f"Nutzen für {name}, Runde %{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=float(dom.r[0]), line=dict(color="gray", dash="dash", width=1), annotation_text="Reservationsnutzen", annotation_position="bottom right")
    fig.update_xaxes(title_text="Runde", dtick=1 if len(view.trace) <= 20 else None)
    fig.update_yaxes(title_text="Nutzen", range=[-0.03, 1.05])
    return _base(fig, 380)


def build_curve_family(r: float):
    """Die Zeitkurven über dem Zeitfortschritt τ: Boulware (β = 0,2), Linear (β = 1), Conceder (β = 5) und Hartnäckig (1)."""
    tau = np.linspace(0, 1, 101)
    fig = go.Figure()
    for name, beta, color in (("Boulware (β = 0,2)", 0.2, C.COLOR_FIRST), ("Linear (β = 1)", 1.0, C.COLOR_FRONT), ("Conceder (β = 5)", 5.0, C.COLOR_SECOND)):
        fig.add_trace(go.Scatter(x=tau, y=1.0 - (1.0 - r) * tau ** (1.0 / beta), mode="lines", line=dict(color=color, width=3), name=name, hovertemplate=name + ": τ = %{x:.2f}, Ziel %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 1], mode="lines", line=dict(color="gray", dash="dot"), name="Hartnäckig", hoverinfo="skip"))
    fig.update_xaxes(title_text="Zeitfortschritt τ = (t − 1) / (T − 1)", range=[0, 1])
    fig.update_yaxes(title_text="Zielnutzen", range=[0, 1.05])
    return _base(fig, 340)


def build_lines(series: dict, x_title: str, y_title: str, yrange=None, categorical=False, height=340, hline=None, colors=None):
    """`series`: Name → (x, y, Fehler oder None). Linien mit Markern, optional mit Fehlerbalken (Standardabweichung über die Seeds); `hline` = (y, Beschriftung) für eine Bezugslinie."""
    fig = go.Figure()
    for name, (xs, ys, err) in series.items():
        e = dict(type="data", array=list(err), visible=True) if err is not None else None
        line = dict(color=colors[name]) if colors and name in colors else None
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines+markers", name=name, error_y=e, line=line, marker=dict(color=colors[name]) if line else None, hovertemplate=f"{name}: %{{x}} | %{{y:.3f}}<extra></extra>"))
    if hline is not None:
        fig.add_hline(y=hline[0], line=dict(color="gray", dash="dot", width=1), annotation_text=hline[1], annotation_position="bottom right")
    fig.update_xaxes(title_text=x_title, type="category" if categorical else None)
    fig.update_yaxes(title_text=y_title, range=yrange)
    return _base(fig, height)


def build_bars(names, means, sds, x_title: str, highlight=None, xrange=None, height=None):
    """Waagerechte Balken, bester oben; `sds`: Standardabweichung über die Seeds als Fehlerbalken."""
    order = np.argsort(means)
    colors = [C.COLOR_FIRST if (highlight is not None and names[i] == highlight) else C.COLOR_FRONT for i in order]
    fig = go.Figure(go.Bar(x=[means[i] for i in order], y=[names[i] for i in order], orientation="h", marker_color=colors,
                           error_x=dict(type="data", array=[sds[i] for i in order], visible=True) if sds is not None else None,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text=x_title, range=xrange)
    return _base(fig, height or max(240, 46 * len(names) + 70), legend_below=False)


def build_heat(matrix, rows, cols, title: str, zmin=None, zmax=None, fmt=".2f"):
    """Matrix als Farbfläche mit Zahlen; `rows` = zuerst bietende Strategie, `cols` = zweite."""
    z = np.array(matrix, dtype=float)
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=rows, colorscale="Viridis", zmin=zmin, zmax=zmax, text=[[f"{v:{fmt}}".replace(".", ",") for v in row] for row in z], texttemplate="%{text}",
                               hovertemplate="zuerst %{y} gegen %{x}: %{z:.3f}<extra></extra>", colorbar=dict(title=title, thickness=12)))
    fig.update_yaxes(autorange="reversed", title_text="bietet zuerst")
    fig.update_xaxes(title_text="bietet als Zweite", side="bottom")
    return _base(fig, 60 * len(rows) + 120, legend_below=False)


def build_stacked(labels, parts: dict, y_title: str, height=360):
    """Gestapelte Balken (Anteile an der Varianz): `parts`: Name → Werte je Beschriftung."""
    fig = go.Figure()
    for name, vals in parts.items():
        fig.add_trace(go.Bar(x=labels, y=vals, name=name, hovertemplate=name + ": %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text=y_title, range=[0, 1.0])
    return _base(fig, height)


def strategy_color(name: str) -> str:
    return {ST.HARTNAECKIG: "#7f7f7f", ST.BOULWARE: C.COLOR_FIRST, ST.LINEAR: C.COLOR_FRONT, ST.CONCEDER: C.COLOR_SECOND, ST.TFT: C.COLOR_NASH, ST.BOULWARE_MODELL: "#1b9e77", ST.ENDSPIEL: "#d95f02"}.get(name, "#999999")
