"""Sweep orchestrator for the HAR / CIFAR-10 / DHCD experiments."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pandas as pd


HERE = Path(__file__).resolve().parent
SIMULATOR = HERE / "fl_simulation_phase3.py"


# Matrix definitions
SEEDS              = [42, 123, 456, 789, 2024]
HEADLINE_METHODS   = ["fedavg", "fedprox", "poc", "oort", "gt_linucb", "abfb"]
DATASETS           = ["har", "dhcd"]
CLIENT_COUNTS      = [10, 50]
PARETO_BUDGETS     = [0, 1, 2, 5, 10, 20]
DEFAULT_BUDGET     = 2
DEFAULT_ROUNDS     = 100
DIRICHLET_ALPHA    = 0.3

# ABFB ablation configs (run on HAR N=50  -  most stressful + cheaper than DHCD)
ABLATION_CONFIGS = [
    # (label,                       extra env vars to inject)
    ("ablation_no_battery_belief",  {"ABFB_NO_BATTERY": "1"}),
    ("ablation_no_channel_belief",  {"ABFB_NO_CHANNEL": "1"}),
    ("ablation_no_compute_belief",  {"ABFB_NO_COMPUTE": "1"}),
    ("ablation_no_coldstart",       {"ABFB_NO_COLDSTART": "1"}),
    ("ablation_tau_fixed_005",      {"ABFB_TAU_FIXED": "0.05"}),
    ("ablation_tau_fixed_05",       {"ABFB_TAU_FIXED": "0.5"}),
    ("ablation_tau_fixed_20",       {"ABFB_TAU_FIXED": "2.0"}),
]


@dataclass
class RunSpec:
    method: str
    dataset: str
    n_clients: int
    budget: int
    seed: int
    rounds: int = DEFAULT_ROUNDS
    label: str = ""        # set for ablations
    env: dict | None = None

    @property
    def run_id(self) -> str:
        base = f"{self.method}-{self.dataset}-n{self.n_clients}-b{self.budget}-s{self.seed}-r{self.rounds}"
        return f"{base}-{self.label}" if self.label else base


def build_headline_matrix() -> List[RunSpec]:
    runs = []
    for method in HEADLINE_METHODS:
        for ds in DATASETS:
            for n in CLIENT_COUNTS:
                for s in SEEDS:
                    runs.append(RunSpec(
                        method=method, dataset=ds, n_clients=n,
                        budget=DEFAULT_BUDGET if method == "abfb" else 0,
                        seed=s,
                    ))
    return runs


def build_pareto_matrix() -> List[RunSpec]:
    runs = []
    for ds in DATASETS:
        for n in CLIENT_COUNTS:
            for b in PARETO_BUDGETS:
                for s in SEEDS:
                    runs.append(RunSpec(
                        method="abfb", dataset=ds, n_clients=n,
                        budget=b, seed=s,
                    ))
    return runs


def build_ablation_matrix() -> List[RunSpec]:
    runs = []
    for (label, env) in ABLATION_CONFIGS:
        for s in SEEDS:
            runs.append(RunSpec(
                method="abfb", dataset="har", n_clients=50,
                budget=DEFAULT_BUDGET, seed=s,
                label=label, env=env,
            ))
    return runs


def build_smoke_matrix() -> List[RunSpec]:
    """4-run smoke: ABFB + GT-LinUCB on HAR-N10 + DHCD-N10, seed 42 only."""
    return [
        RunSpec(method="abfb",      dataset="har",  n_clients=10, budget=2, seed=42, rounds=20),
        RunSpec(method="gt_linucb", dataset="har",  n_clients=10, budget=0, seed=42, rounds=20),
        RunSpec(method="abfb",      dataset="dhcd", n_clients=10, budget=2, seed=42, rounds=20),
        RunSpec(method="gt_linucb", dataset="dhcd", n_clients=10, budget=0, seed=42, rounds=20),
    ]


# Run execution
def out_dir_for(spec: RunSpec) -> Path:
    base = HERE / f"phase3_results_{spec.dataset}_n{spec.n_clients}"
    if spec.label:
        return base / f"{spec.label}_s{spec.seed}"
    return base / f"{spec.method}_b{spec.budget}_s{spec.seed}"


def already_done(spec: RunSpec) -> bool:
    return (out_dir_for(spec) / "summary.json").exists()


def run_one(spec: RunSpec, verbose: bool = False, max_retries: int = 1,
            timeout_s: int = 4 * 3600) -> dict:
    """Invoke the simulator as a subprocess. Return summary dict.

    Default per-run timeout is 4 hours  -  FedProx on DHCD-N50 + 100 rounds
    can take ~2–3 hours, so anything less is too tight.
    """
    if already_done(spec):
        return {"status": "skip_existing", "run_id": spec.run_id}

    cmd = [
        sys.executable, str(SIMULATOR),
        "--method",     spec.method,
        "--dataset",    spec.dataset,
        "--n_clients",  str(spec.n_clients),
        "--rounds",     str(spec.rounds),
        "--budget",     str(spec.budget),
        "--seed",       str(spec.seed),
        "--dirichlet_alpha", str(DIRICHLET_ALPHA),
    ]
    if not verbose:
        cmd.append("--quiet")

    import os
    env = os.environ.copy()
    if spec.env:
        env.update(spec.env)

    t0 = time.time()
    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=timeout_s,
                check=True,
            )
            elapsed = time.time() - t0
            return {
                "status": "ok",
                "run_id": spec.run_id,
                "wall_s": elapsed,
                "attempt": attempt + 1,
                "out_dir": str(out_dir_for(spec)),
            }
        except subprocess.CalledProcessError as e:
            if attempt < max_retries:
                continue
            return {
                "status": "fail",
                "run_id": spec.run_id,
                "returncode": e.returncode,
                "stderr_tail": (e.stderr or "")[-500:],
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "run_id": spec.run_id,
                "elapsed_s": time.time() - t0,
            }


# Aggregation
def aggregate_summaries(stage_name: str = "all") -> Path:
    """Walk all phase3_results_* dirs, collect summary.json files."""
    rows = []
    for run_dir in sorted(HERE.glob("phase3_results_*/*/summary.json")):
        try:
            with open(run_dir) as f:
                d = json.load(f)
            d["run_dir"] = str(run_dir.parent)
            rows.append(d)
        except Exception:
            continue
    df = pd.DataFrame(rows)
    out = HERE / f"final_summary_{stage_name}.csv"
    df.to_csv(out, index=False)
    print(f"[Sweep] Aggregated {len(df)} runs → {out}")
    return out


# Main
STAGES = {
    "smoke":     build_smoke_matrix,
    "headline":  build_headline_matrix,
    "pareto":    build_pareto_matrix,
    "ablation":  build_ablation_matrix,
    "all":       lambda: build_headline_matrix() + build_pareto_matrix()
                          + build_ablation_matrix(),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES.keys()))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--timeout_h", type=float, default=4.0,
                    help="Per-run timeout in hours (default 4.0).")
    return ap.parse_args()


def main():
    args = parse_args()
    runs = STAGES[args.stage]()
    print(f"\n[Sweep] Stage='{args.stage}' → {len(runs)} runs scheduled")

    if args.dry_run:
        for r in runs:
            print(f"  {r.run_id}")
        return

    pending = [r for r in runs if not already_done(r)]
    skipped = len(runs) - len(pending)
    timeout_s = int(args.timeout_h * 3600)
    print(f"[Sweep] Skipping {skipped} already-completed runs; running {len(pending)}.")
    print(f"[Sweep] Per-run timeout: {args.timeout_h:.1f} h ({timeout_s} s)")

    results = []
    t_start = time.time()

    if args.workers == 1:
        for i, spec in enumerate(pending, 1):
            print(f"\n[Sweep] ({i}/{len(pending)}) {spec.run_id}")
            res = run_one(spec, verbose=args.verbose, timeout_s=timeout_s)
            results.append(res)
            print(f"  → {res['status']}")
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, s, args.verbose, 1, timeout_s): s
                       for s in pending}
            for i, fut in enumerate(cf.as_completed(futures), 1):
                res = fut.result()
                results.append(res)
                print(f"[Sweep] ({i}/{len(pending)}) {res['run_id']} → {res['status']}")

    elapsed_h = (time.time() - t_start) / 3600
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] in {"fail", "timeout"})
    print(f"\n[Sweep] Done: {n_ok} ok, {n_fail} failed, in {elapsed_h:.2f} h")

    # Persist run log
    log_path = HERE / f"sweep_log_{args.stage}_{int(time.time())}.json"
    with open(log_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"[Sweep] Run log → {log_path}")

    # Aggregate summaries
    aggregate_summaries(args.stage)


if __name__ == "__main__":
    main()
