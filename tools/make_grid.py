"""Rechnet alle Messreihen neu (Turnier-Raster, Endspiel, Position, Gegnermodell, Zerlegung, Robustheit, Lernender) und schreibt data/messreihen.json samt Erzeuger-Hash.

Aufruf (im Demo-Ordner, mit dem Test-venv):  python tools/make_grid.py [--workers 14] [--only turnier,position]
Ändert sich eine der Quelldateien aus `vha_experiments.HASHED_FILES` oder ein Messparameter in vha_constants.py, stimmt der Hash nicht mehr (tests/test_measurements.py schlägt an, die App zeigt
die Reihen dann nicht); danach diesen Aufruf wiederholen. `--only` rechnet nur die genannten Auftragsarten und führt sie mit der vorhandenen Datei zusammen (nur sinnvoll, solange der Hash
unverändert ist; ein Lauf ohne `--only` ersetzt alles). Jeder Auftrag hat feste Seeds aus seiner Bezeichnung: die Reihen sind bei gleichem Code Zahl für Zahl dieselben, auch bei anderer
Zahl von Prozessen. Dauer: wenige Minuten auf 14 Kernen (die Trainingsläufe des Lernenden dominieren)."""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vha_experiments as XP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    start = time.time()
    jobs = XP.all_jobs()
    only = {x for x in args.only.split(",") if x}
    results = {}
    if only:
        old = json.loads(XP.DATA_FILE.read_text(encoding="utf-8"))
        if old.get("generator_hash") != XP.generator_hash():
            sys.exit("Die vorhandene Datei gehört zu einem anderen Rechencode: ohne --only neu rechnen.")
        results = {k: v for k, v in old["jobs"].items() if k.split("|")[0] not in only}
        jobs = [j for j in jobs if j[0] in only]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for key, res in ex.map(XP.run_job, jobs, chunksize=1):
            results[key] = res
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} Aufträge, {time.time() - start:.0f} s", flush=True)
    XP.write(XP.assemble(results))
    print(f"geschrieben: {XP.DATA_FILE}  ({time.time() - start:.0f} s, Hash {XP.generator_hash()}, {len(jobs)} Aufträge neu, {len(results)} gesamt)")


if __name__ == "__main__":
    main()
