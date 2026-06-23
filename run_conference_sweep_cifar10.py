"""Conference paper CIFAR-10 sweep: 21 runs (5 methods x 3 seeds + Pareto)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE / "fl_simulation_phase3.py"


def build_matrix() -> list[dict]:
    """Yield the full run matrix as a flat list of CLI-arg dicts."""
    matrix: list[dict] = []

    methods_headline = ["fedavg", "poc", "oort", "gt_linucb", "abfb"]
    seeds = [0, 1, 2]
    n_clients = 10
    rounds = 100

    # Headline runs  -  ABFB at B=2
    for m in methods_headline:
        for s in seeds:
            matrix.append({
                "method": m, "dataset": "cifar10", "n_clients": n_clients,
                "rounds": rounds, "seed": s,
                "budget": 2 if m == "abfb" else 0,
            })

    # Pareto runs for ABFB only  -  B=0 and B=N
    for b in (0, n_clients):
        for s in seeds:
            matrix.append({
                "method": "abfb", "dataset": "cifar10", "n_clients": n_clients,
                "rounds": rounds, "seed": s, "budget": b,
            })

    return matrix


def run_label(cfg: dict) -> str:
    return (f"{cfg['method']:>9}_b{cfg['budget']:>2}_n{cfg['n_clients']}_"
            f"s{cfg['seed']}_{cfg['dataset']}")


def out_dir(cfg: dict) -> Path:
    return (HERE / f"phase3_results_{cfg['dataset']}_n{cfg['n_clients']}"
                 / f"{cfg['method']}_b{cfg['budget']}_s{cfg['seed']}")


def already_done(cfg: dict) -> bool:
    """A run is considered complete if its summary.json exists."""
    return (out_dir(cfg) / "summary.json").exists()


def run_one(cfg: dict, log_dir: Path) -> int:
    """Spawn one simulator process; return its exit code."""
    cmd = [
        sys.executable, str(SIM),
        "--dataset",         cfg["dataset"],
        "--method",          cfg["method"],
        "--n_clients",       str(cfg["n_clients"]),
        "--rounds",          str(cfg["rounds"]),
        "--seed",            str(cfg["seed"]),
        "--budget",          str(cfg["budget"]),
        "--dirichlet_alpha", "0.3",
        "--q_agent",         "linfa",
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    label = run_label(cfg)
    log_path = log_dir / f"{label}.log"
    print(f"  → {label}  (log: {log_path.name})")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=HERE)
    elapsed = time.time() - t0
    print(f"    exit={proc.returncode}  elapsed={elapsed/60:.1f} min")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true",
                        help="Print plan and exit without running.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose summary.json already exists.")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method filter (e.g. abfb,gt_linucb).")
    args = parser.parse_args()

    matrix = build_matrix()
    if args.methods:
        wanted = set(args.methods.split(","))
        matrix = [c for c in matrix if c["method"] in wanted]

    print(f"Conference sweep (CIFAR-10)  -  {len(matrix)} configured runs")
    print(f"Project root: {HERE}")
    print(f"Simulator   : {SIM}")
    print("Run plan:")
    for i, cfg in enumerate(matrix, 1):
        flag = "(DONE)" if already_done(cfg) else "      "
        print(f"  {i:3d}. {flag} {run_label(cfg)}")

    if args.dry:
        print("\nDry run  -  exiting.")
        return 0

    log_dir = HERE / "logs_conference_sweep_cifar10"

    failed: list[str] = []
    skipped: list[str] = []
    t_start = time.time()
    for i, cfg in enumerate(matrix, 1):
        label = run_label(cfg)
        if args.resume and already_done(cfg):
            print(f"\n[{i}/{len(matrix)}] SKIP (already done): {label}")
            skipped.append(label)
            continue
        print(f"\n[{i}/{len(matrix)}] {label}")
        rc = run_one(cfg, log_dir)
        if rc != 0:
            failed.append(label)

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Sweep complete in {elapsed/60:.1f} min")
    print(f"  Total runs   : {len(matrix)}")
    print(f"  Succeeded    : {len(matrix) - len(failed) - len(skipped)}")
    print(f"  Skipped (done): {len(skipped)}")
    print(f"  Failed       : {len(failed)}")
    if failed:
        print("Failed labels:")
        for f in failed:
            print(f"  - {f}")
    print("=" * 70)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
