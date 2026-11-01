"""Gemeinsame Fixtures: die Mini-Domäne, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Erwartungswerte stehen als Rechnung dabei."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vha_domain import Domain  # noqa: E402


@pytest.fixture
def mini():
    """Zwei Punkte mit je drei Werten, Gewichte (0,6; 0,4) für Position 0 und (0,3; 0,7) für Position 1, r = 0,2 für beide.
    Wertlisten (Position 0: 1; 0,5; 0 je Punkt, Position 1: 0; 0,5; 1 je Punkt). Bündel b = 3·x₀ + x₁ mit Werten x₀, x₁ ∈ {0, 1, 2}.
    Nutzen von Hand, Position 0: u₀(b) = 0,6·v(x₀) + 0,4·v(x₁) mit v = (1; 0,5; 0):
      b0 (0,0) = 1,0   b1 (0,1) = 0,6 + 0,2 = 0,8   b2 (0,2) = 0,6   b3 (1,0) = 0,3 + 0,4 = 0,7   b4 (1,1) = 0,3 + 0,2 = 0,5   b5 (1,2) = 0,3   b6 (2,0) = 0,4   b7 (2,1) = 0,2   b8 (2,2) = 0.
    Position 1: u₁(b) = 0,3·v'(x₀) + 0,7·v'(x₁) mit v' = (0; 0,5; 1):
      b0 = 0   b1 = 0,35   b2 = 0,7   b3 = 0,15   b4 = 0,5   b5 = 0,85   b6 = 0,3   b7 = 0,65   b8 = 1,0.
    Die Tabelle `tie` (Gleichstand: kleinster Wert zuerst) ist 0,00; 0,01; …; 0,08, also gewinnt bei Gleichstand der kleinere Bündelindex."""
    w = [[0.6, 0.4], [0.3, 0.7]]
    v = [[[1, .5, 0], [1, .5, 0]], [[0, .5, 1], [0, .5, 1]]]
    return Domain(w, v, [0.2, 0.2], np.arange(9) / 100.0)


MINI_U0 = [1.0, .8, .6, .7, .5, .3, .4, .2, 0.0]
MINI_U1 = [0.0, .35, .7, .15, .5, .85, .3, .65, 1.0]
