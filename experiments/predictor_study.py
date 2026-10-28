"""Offline study of 1-hour zone-temperature predictors on recorded closed-loop traces (H1).

Traces come from the control policy (HRC-PI with kappa = 0, bit-identical to the PI
controller) on the *tune* scenarios only, so choosing predictor settings here never
looks at validation data. Each predictor is replayed step by step exactly as it runs
inside the controller (same online RLS, same delayed target), and scored by the RMSE
of its 1 h forecast from day 2 onwards, against persistence (zero drift).

    python experiments/predictor_study.py results/traces results/predictor_study.csv
"""
from __future__ import annotations

import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hrc_hvac.reservoir import make_predictor  # noqa: E402

H = 12
SKIP = 288  # first day excluded from scoring (warm-up)


def replay(U: np.ndarray, T: np.ndarray, kind: str, **kw) -> dict:
    steps, Z, D = U.shape
    p = make_predictor(kind, Z, D, horizon=H, **kw)
    preds = np.full((steps, Z), np.nan)
    for t in range(steps):
        preds[t] = p.step(U[t].astype(np.float64), T[t].astype(np.float64))
    # prediction made at t targets t+H
    e = T[SKIP + H:] - preds[SKIP: steps - H]
    ep = T[SKIP + H:] - T[SKIP: steps - H]
    rmse, prmse = float(np.sqrt(np.mean(e ** 2))), float(np.sqrt(np.mean(ep ** 2)))
    return {"rmse": rmse, "persist_rmse": prmse, "skill": 1.0 - rmse / prmse}


def grid():
    for delta, lam in itertools.product((0.001, 0.01, 0.1, 1.0), (0.9995, 0.9999, 0.99999)):
        yield "ar", dict(delta=delta, lam=lam)
        for size, leak, scale, seed in itertools.product((50, 150), (0.3, 1.0), (0.3, 1.0), (0,)):
            yield "hrc", dict(delta=delta, lam=lam, size=size, leak=leak, input_scale=scale, seed=seed)


_DATA: dict = {}


def _eval(job):
    kind, kw = job
    t0 = time.time()
    res = [replay(d["U"], d["T"], kind, **kw) for d in _DATA.values()]
    return dict(kind=kind, **{k: kw.get(k, "") for k in ("delta", "lam", "size", "leak", "input_scale", "seed")},
                rmse=np.mean([r["rmse"] for r in res]), persist_rmse=np.mean([r["persist_rmse"] for r in res]),
                skill=np.mean([r["skill"] for r in res]), skill_min=np.min([r["skill"] for r in res]),
                sec=round(time.time() - t0, 1))


def main(trace_dir: str, out_csv: str, workers: int = 4) -> None:
    import multiprocessing as mp
    files = sorted(Path(trace_dir).glob("*.npz"))
    for f in files:
        d = np.load(f)
        _DATA[f.name.split("__hrc")[0]] = {"U": d["U"], "T": d["T"]}
    print(f"{len(_DATA)} traces: {list(_DATA)}", flush=True)
    jobs = [("persistence", {})] + list(grid())
    rows = []
    with mp.get_context("fork").Pool(workers) as pool:
        for row in pool.imap(_eval, jobs):
            rows.append(row)
            print(row, flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    import os
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
