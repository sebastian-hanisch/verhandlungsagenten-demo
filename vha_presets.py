"""Permalink (Begrenzen), Voreinstellungen und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

import vha_constants as C
import vha_strategies as ST
from vha_scenario import Settings


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None      # bei Auswahlreglern: die erlaubten Werte


B = C.BASE
STRATEGY_CHOICES = tuple(ST.POOL)
SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, B["n"], C.N_MIN, C.N_MAX),
    "m_slider": SettingSpec("m", int, B["m"], C.M_MIN, C.M_MAX),
    "s_slider": SettingSpec("s", float, B["s"], C.S_MIN, C.S_MAX),
    "g_slider": SettingSpec("g", float, B["g"], C.G_MIN, C.G_MAX),
    "r_slider": SettingSpec("r", float, B["r"], C.R_MIN, C.R_MAX),
    "t_slider": SettingSpec("T", int, B["T"], C.T_MIN, C.T_MAX),
    "band_slider": SettingSpec("band", float, B["band"], C.BAND_MIN, C.BAND_MAX),
    "rule_select": SettingSpec("regel", str, B["rule"], choices=C.RULE_CHOICES),
    "first_select": SettingSpec("erster", str, B["first"], choices=STRATEGY_CHOICES),
    "second_select": SettingSpec("zweiter", str, B["second"], choices=STRATEGY_CHOICES),
    "seed_input": SettingSpec("seed", int, B["seed"], 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "m": "m_slider", "s": "s_slider", "g": "g_slider", "r": "r_slider", "T": "t_slider", "band": "band_slider", "rule": "rule_select",
               "first": "first_select", "second": "second_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf erlaubte Werte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.choices is not None:
                if value not in spec.choices:
                    continue
            else:
                if value != value:      # NaN
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def settings_from_state() -> Settings:
    """Die Einstellungen aus dem Session State (vor dem Zeichnen der Regler gesetzt durch Voreinstellung, Permalink oder Standard)."""
    g = st.session_state.get
    d = {k: g(k, SETTING_SPECS[k].default) for k in SETTING_SPECS}
    return Settings(n=int(d["n_slider"]), m=int(d["m_slider"]), s=float(d["s_slider"]), g=float(d["g_slider"]), r=float(d["r_slider"]), T=int(d["t_slider"]),
                    band=float(d["band_slider"]), rule=str(d["rule_select"]), first=str(d["first_select"]), second=str(d["second_select"]), seed=int(d["seed_input"]))


def aktives_preset():
    """Name der Voreinstellung, deren Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (eine Voreinstellung ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; die aktive Voreinstellung ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
