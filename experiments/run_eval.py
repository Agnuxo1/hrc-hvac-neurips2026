"""Run policies on official-simulator scenarios in parallel and append results to a CSV.

Each job = (policy spec, scenario). Results are appended row by row, so an interrupted
run resumes where it stopped (rows already present are skipped).

    python experiments/run_eval.py --kit /path/to/starter-kit --roots ROOT1 ROOT2 \
        --scenarios experiments/splits/validation.txt --policies zt_best hrc:k=0.5:s=0 \
        --steps 8640 --workers 10 --out results/validation.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import multiprocessing as mp
try:
    import resource
except ImportError:  # Windows
    resource = None
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RC_KEYS = ("normalized_productivity_regret", "normalized_energy_cost", "normalized_carbon_emission")
COLS = ["scenario", "policy", "steps", "avg_reward", "score", "total_reward", "comfort_regret",
        "energy_cost", "carbon", "pred_rmse", "persist_rmse", "gate_frac", "ff_mean_abs",
        "fallbacks", "sec", "s_per_step", "max_rss_mb", "error"]
_G: dict = {}


def parse_spec(spec: str) -> dict:
    """'hrc:k=0.5:s=1' -> {'kind': 'hrc', 'k': 0.5, 's': 1}."""
    parts = spec.split(":")
    out = {"kind": parts[0]}
    for p in parts[1:]:
        k, v = p.split("=")
        out[k] = float(v) if k in ("k", "gate", "drift", "leak", "radius", "lam", "delta", "scale") else v
    return out


def policy_class(spec: str):
    d = parse_spec(spec)
    kind = d["kind"]
    if kind == "pid_water":
        return _G["ev"].POLICY_MAP["pid_water"]
    from hrc_hvac.zone_pi import ZTBestPolicy
    if kind == "zt_best":
        return ZTBestPolicy
    from hrc_hvac.hrc_policy import make_policy_class
    if kind in ("hrc", "ar", "persistence"):
        extra = {}
        if "gate" in d: extra["gate_ratio"] = d["gate"]
        if "drift" in d: extra["max_drift_k"] = d["drift"]
        if "n" in d: extra["res_size"] = int(d["n"])
        if "leak" in d: extra["res_leak"] = d["leak"]
        if "radius" in d: extra["res_radius"] = d["radius"]
        if "lam" in d: extra["rls_lambda"] = d["lam"]
        if "delta" in d: extra["rls_delta"] = d["delta"]
        if "scale" in d: extra["res_input_scale"] = d["scale"]
        if d.get("trace") == "1": extra["trace"] = True
        return make_policy_class(kappa=d.get("k", 0.5), predictor=kind, seed=int(d.get("s", 0)), **extra)
    raise ValueError(f"unknown policy spec {spec!r}")


def find_scenario(name: str) -> Path:
    for root in _G["roots"]:
        p = Path(root) / name
        if (p / "config.gin").exists():
            return p
    raise FileNotFoundError(name)


def _init(kit: str, roots: list[str], steps: int, out: str) -> None:
    os.chdir(kit)
    sys.path.insert(0, kit)
    sys.path.insert(0, str(REPO))
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
    _G.update(ev=ev, Tracker=SimulationTracker, roots=roots, steps=steps, out=out)


def _mem_available_mb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("inf")


def _job(args):
    scenario, spec = args
    # Shared server: never start a simulation that could push other users' jobs into OOM.
    while _mem_available_mb() < float(os.environ.get("SB_MIN_FREE_MB", "2500")):
        time.sleep(30)
    t0 = time.time()
    row = {"scenario": scenario, "policy": spec, "steps": _G["steps"]}
    try:
        ev = _G["ev"]
        sdir = find_scenario(scenario)
        env = ev.load_hybrid_environment(sdir / "config.gin")
        tracker = _G["Tracker"](env, vmin=280, vmax=310, capture_snapshots=False)
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0  # same as the kit's evaluate.py
        meta = ev.create_building_meta_data(env, sdir)
        pol = ev.initialize_policy(policy_class(spec), meta, env.action_spec())
        s = ev.run_policy(env, tracker, pol, _G["steps"])
        rc = tracker.reward_components or []
        sums = [float(sum(float(d.get(k, 0.0)) for d in rc)) for k in RC_KEYS]
        row.update(avg_reward=s["mean_reward"], score=abs(s["mean_reward"]), total_reward=s["total_reward"],
                   comfort_regret=sums[0], energy_cost=sums[1], carbon=sums[2],
                   s_per_step=s["seconds_per_step"], steps=s["num_steps"])
        if hasattr(pol, "hrc_stats"):
            st = pol.hrc_stats
            pr, pp = pol.predictor_rmse()
            if getattr(pol, "TRACE", False) and pol.trace_u:
                import numpy as np
                tdir = Path(_G["out"]).parent / "traces"
                tdir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tdir / f"{scenario.replace('/', '__')}__{spec.replace(':', '_')}.npz",
                                    U=np.stack(pol.trace_u), T=np.stack(pol.trace_T))
            row.update(pred_rmse=pr, persist_rmse=pp, fallbacks=st["fallbacks"],
                       gate_frac=st["gate_steps"] / max(1, st["steps"]),
                       ff_mean_abs=st["ff_abs_sum"] / max(1, st["ff_n"]))
    except Exception as exc:  # recorded, never hidden
        row["error"] = f"{type(exc).__name__}: {exc} | {traceback.format_exc()[-400:]!r}"
    row["sec"] = round(time.time() - t0, 1)
    if resource is not None:
        row["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--scenarios", required=True, help="text file, one building/weather per line")
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    scen = [l.strip() for l in open(a.scenarios) if l.strip() and not l.startswith("#")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["scenario"], r["policy"]) for r in csv.DictReader(f) if not r.get("error")}
    jobs = [(s, p) for p in a.policies for s in scen if (s, p) not in done]
    # longest scenarios first (generated buildings are the slowest)
    jobs.sort(key=lambda j: 0 if j[0].startswith("building_9") else 1)
    print(f"{len(jobs)} jobs ({len(done)} already done), {a.workers} workers", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, mp.get_context("fork").Pool(
            a.workers, initializer=_init, initargs=(a.kit, a.roots, a.steps, str(out.resolve())), maxtasksperchild=1) as pool:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for row in pool.imap_unordered(_job, jobs):
            w.writerow(row)
            f.flush()
            print(time.strftime("%H:%M:%S"), row["scenario"], row["policy"], row.get("score"),
                  row.get("sec"), (row.get("error") or "")[:120], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
