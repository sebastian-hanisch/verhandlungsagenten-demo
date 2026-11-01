"""Die vorgerechneten Messreihen (data/messreihen.json) gehören zum aktuellen Rechencode und werden von der Rechnung reproduziert.

Eigene Datei, weil jede Änderung an einer der Quelldateien aus `HASHED_FILES` oder an den Messparametern in vha_constants den Erzeuger-Hash ändert. Reproduziert wird plattformsicher: einzelne Aufträge
neu gerechnet und mit dem gespeicherten Ergebnis verglichen (die Werte sind auf sechs Stellen gerundet gespeichert; eine kleine Marge bleibt wegen unterschiedlicher Summationsreihenfolge)."""

import json

import numpy as np
import pytest

import vha_constants as C
import vha_experiments as XP


@pytest.fixture(scope="module")
def data():
    return json.loads(XP.DATA_FILE.read_text(encoding="utf-8"))


def _flat(x):
    if isinstance(x, dict):
        return [v for k in sorted(x) for v in _flat(x[k])]
    if isinstance(x, list):
        return [v for e in x for v in _flat(e)]
    if isinstance(x, str):
        return []                      # Namen (robust): getrennt in test_a_stored_job_names_are_reproduced geprüft
    return [np.nan if x is None else float(x)]


def test_file_matches_the_generator_hash(data):
    assert data["generator_hash"] == XP.generator_hash(), "Quellcode geändert: python tools/make_grid.py ausführen"
    assert XP.load() is not None


def test_load_returns_none_for_a_stale_hash_and_for_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(XP, "generator_hash", lambda: "0000000000000000")
    assert XP.load() is None
    monkeypatch.undo()
    monkeypatch.setattr(XP, "DATA_FILE", tmp_path / "fehlt.json")
    assert XP.load() is None


def test_hash_ignores_the_presets_part_of_the_constants():
    text = (XP.ROOT / "vha_constants.py").read_text(encoding="utf-8")
    before, after = text.split(XP.HASH_END)
    assert "TURNIER_TS" in before and "LEARN_EPISODES" in before and "PRESETS = {" in after and "PRESETS = {" not in before


def test_every_job_is_stored_and_there_are_no_extra_ones(data):
    jobs = XP.all_jobs()
    assert len(jobs) == len({XP.job_key(j) for j in jobs}) == 480
    assert set(data["jobs"]) == {XP.job_key(j) for j in jobs}
    assert data["seeds"] == list(C.SEEDS)


def test_job_counts_by_kind(data):
    from collections import Counter
    kinds = Counter(j[0] for j in XP.all_jobs())
    assert kinds == {"turnier": 36, "regelaus": 4, "endspiel": 8, "position": 32, "lernmodell": 40, "wirkmodell": 36, "zerlegung": 192, "akzeptanz": 36, "robust": 72, "lerner": 12,
                     "lerner_basis": 4, "lerner_aus": 8}


def test_no_domain_was_skipped_in_the_tournament_and_position_domains_are_shared(data):
    assert all(data["jobs"][XP.job_key(("turnier", T, float(s), sd))]["skipped"] == 0 for T in C.TURNIER_TS for s in C.TURNIER_SS for sd in C.SEEDS)
    # dieselben Domänen für alle Fristen: die Zahl verworfener Domänen ist je (s, Seed) gleich
    for s in C.TURNIER_SS:
        for sd in C.SEEDS:
            assert len({data["jobs"][XP.job_key(("turnier", T, float(s), sd))]["skipped"] for T in C.TURNIER_TS}) == 1


@pytest.mark.parametrize("job", [("turnier", 6, 0.0, 1), ("turnier", 6, 1.0, 3), ("regelaus", 2), ("position", 6, 1, 1), ("position", 21, 0, 4), ("lernmodell", "Conceder", 0.5, 2),
                                 ("wirkmodell", "Linear", "haeufig", 1), ("zerlegung", "Tit-for-Tat", "Zeitplan", "Orakel", 4), ("akzeptanz", 6, 3), ("robust", 14, 2)])
def test_a_stored_job_is_reproduced_by_the_code(data, job):
    key, fresh = XP.run_job(job)
    if "names" in fresh:
        assert fresh["names"] == data["jobs"][key]["names"]
    got, want = np.array(_flat(fresh)), np.array(_flat(data["jobs"][key]))
    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-5, equal_nan=True), key


def test_a_short_learning_job_is_reproduced_exactly_in_structure(data):
    """Der Lernende (5.000 Episoden) wäre zu teuer für die Suite; geprüft wird der Aufbau des gespeicherten Ergebnisses und eine kurze Trainingsfolge ist deterministisch (test_learning)."""
    res = data["jobs"][XP.job_key(("lerner", 20000, 1))]
    assert set(res["ref"]) == {"Hartnäckig", "Boulware", "Linear", "Conceder", "Tit-for-Tat (relativ)", "Boulware + Modell"} and set(res["transfer"]) == {t[0] for t in XP.TRANSFER}
    assert all(0 <= v[0] <= 1 and 0 <= v[1] <= 1 for v in res["ref"].values())
    assert "transfer" not in data["jobs"][XP.job_key(("lerner", 5000, 1))]


def test_pair_matrices_have_the_documented_shape(data):
    M = np.array(data["jobs"][XP.job_key(("turnier", 20, 0.5, 1))]["M"], dtype=float)
    assert M.shape == (6, 6, 7) and np.isnan(M[:, :, 6]).sum() <= 2
    M9 = np.array(data["jobs"][XP.job_key(("endspiel", 20, 1))]["M"], dtype=float)
    assert M9.shape == (9, 9, 7)
