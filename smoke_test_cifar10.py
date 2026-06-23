"""Smoke test for the CIFAR-10 pipeline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SIM = HERE / "fl_simulation_phase3.py"


def run_sim(method: str, seed: int, rounds: int, n_clients: int, budget: int) -> dict:
    """Spawn the simulator as a subprocess; parse its summary.json result."""
    cmd = [
        sys.executable, str(SIM),
        "--dataset",         "cifar10",
        "--method",          method,
        "--n_clients",       str(n_clients),
        "--rounds",          str(rounds),
        "--seed",            str(seed),
        "--budget",          str(budget),
        "--dirichlet_alpha", "0.3",
        "--q_agent",         "linfa",
    ]
    print(f"  Running: {method} seed={seed} rounds={rounds} N={n_clients} B={budget}")
    t0 = time.time()
    # Show child stdout/stderr live so the user sees progress
    proc = subprocess.run(cmd, cwd=HERE)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return {"method": method, "elapsed_s": elapsed,
                "ok": False, "error": f"exit code {proc.returncode}"}

    # Find this run's summary.json
    out_dir = (HERE / f"phase3_results_cifar10_n{n_clients}"
                    / f"{method}_b{budget}_s{seed}")
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return {"method": method, "elapsed_s": elapsed, "ok": False,
                "error": f"summary.json not found at {summary_path}"}

    with open(summary_path) as f:
        summary = json.load(f)
    summary["elapsed_s"] = elapsed
    summary["ok"] = True
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=4)
    args = parser.parse_args()

    print("=" * 70)
    print(f"CIFAR-10 smoke test: {args.rounds} rounds, {args.clients} clients")
    device = ('CUDA (' + torch.cuda.get_device_name(0) + ')'
              if torch.cuda.is_available() else 'CPU')
    print(f"Device: {device}")
    print("=" * 70)

    # Step 1: Loader sanity
    print("\n[Step 1] Loader sanity check ...")
    from cifar10_loader import build_cifar10_loaders
    loaders, test_loader, ns = build_cifar10_loaders(
        num_clients=args.clients, batch_size=32, seed=0, dirichlet_alpha=0.3)
    print(f"  Train clients: {len(loaders)}, sample counts: {ns}")
    print(f"  Test split:    {len(test_loader.dataset)} samples")
    x, y = next(iter(loaders[0]))
    print(f"  Batch: x={tuple(x.shape)}, y={tuple(y.shape)}, "
          f"dtype={x.dtype}, range=[{x.min():.3f}, {x.max():.3f}]")

    # Step 2: Model sanity
    print("\n[Step 2] CIFAR10Net forward check ...")
    from cifar10_net import CIFAR10Net, cifar10net_param_count
    net = CIFAR10Net()
    n_params = cifar10net_param_count()
    print(f"  Params: {n_params:,} ({n_params*4/1024:.1f} KB FP32)")
    logits = net(x)
    print(f"  Output shape: {tuple(logits.shape)}  (expected (32, 10))")
    assert logits.shape == (x.shape[0], 10)

    # Step 3: Simulator end-to-end on GT-LinUCB
    print("\n[Step 3] Mini-sim: GT-LinUCB on CIFAR-10 (subprocess) ...")
    r_gt = run_sim("gt_linucb", seed=0, rounds=args.rounds,
                   n_clients=args.clients, budget=0)
    if not r_gt.get("ok"):
        print(f"  FAILED: {r_gt.get('error')}")
        return 1
    print(f"  Final accuracy: {r_gt.get('final_accuracy', 0):.2f}%")
    print(f"  Total energy:   {r_gt.get('cumulative_energy_j', 0):.1f} J")
    print(f"  Elapsed:        {r_gt['elapsed_s']:.1f} s")

    # Step 4: Simulator end-to-end on ABFB
    print("\n[Step 4] Mini-sim: ABFB on CIFAR-10 (subprocess) ...")
    r_abfb = run_sim("abfb", seed=0, rounds=args.rounds,
                     n_clients=args.clients, budget=2)
    if not r_abfb.get("ok"):
        print(f"  FAILED: {r_abfb.get('error')}")
        return 1
    print(f"  Final accuracy: {r_abfb.get('final_accuracy', 0):.2f}%")
    print(f"  Total energy:   {r_abfb.get('cumulative_energy_j', 0):.1f} J")
    print(f"  Elapsed:        {r_abfb['elapsed_s']:.1f} s")

    # Step 5: Sanity ranges
    print("\n[Step 5] Per-round energy + accuracy sanity ...")
    per_round_gt   = r_gt.get('cumulative_energy_j', 0)   / max(1, args.rounds)
    per_round_abfb = r_abfb.get('cumulative_energy_j', 0) / max(1, args.rounds)
    print(f"  GT-LinUCB per-round: {per_round_gt:.1f} J  (target: 1000-20000 J)")
    print(f"  ABFB     per-round: {per_round_abfb:.1f} J  (target: 1000-20000 J)")
    # NOTE: final_accuracy is in percent (e.g. 12.34), not [0,1]
    acc_gt   = r_gt.get('final_accuracy',   0)
    acc_abfb = r_abfb.get('final_accuracy', 0)
    print(f"  GT-LinUCB final acc: {acc_gt:.2f}%  (target: > 10% = chance)")
    print(f"  ABFB     final acc: {acc_abfb:.2f}%  (target: > 10% = chance)")

    ok = (
        acc_gt   > 10.0 and
        acc_abfb > 10.0 and
        1000 <= per_round_gt   <= 20000 and
        1000 <= per_round_abfb <= 20000
    )

    print("\n" + "=" * 70)
    if ok:
        print("SMOKE TEST PASSED ✓")
        print("=" * 70)
        print("\nNext step:")
        print("  python run_conference_sweep_cifar10.py --dry      # preview")
        print("  python run_conference_sweep_cifar10.py            # full run (~5–8 h)")
        return 0
    print("SMOKE TEST FAILED ✗  -  check accuracy / energy ranges above")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
