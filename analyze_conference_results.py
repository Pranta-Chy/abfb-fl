"""Aggregates HAR + CIFAR-10 per-seed results into the paper tables and figures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "conference_paper_outputs"
OUT_DIR.mkdir(exist_ok=True)

CIFAR_ROOT = HERE / "phase3_results_cifar10_n10"
# HAR results live under results/phase3_results_har_n10 (existing 5-seed sweep)
HAR_ROOT   = HERE / "results" / "phase3_results_har_n10"
if not HAR_ROOT.exists():
    HAR_ROOT = HERE / "phase3_results_har_n10"

METHODS_HEADLINE = ['fedavg', 'poc', 'oort', 'gt_linucb', 'abfb']
ABFB_BUDGETS_CIFAR = [0, 2, 10]


# Loaders
def load_summary(method: str, budget: int, seed: int, dataset_root: Path) -> dict | None:
    p = dataset_root / f"{method}_b{budget}_s{seed}" / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_round_csv(method: str, budget: int, seed: int, dataset_root: Path) -> pd.DataFrame | None:
    run_dir = dataset_root / f"{method}_b{budget}_s{seed}"
    if not run_dir.exists():
        return None
    csvs = list(run_dir.glob(f"{method}_rounds.csv"))
    if not csvs:
        csvs = list(run_dir.glob("*_rounds.csv"))
    if not csvs:
        return None
    return pd.read_csv(csvs[0])


def discover_seeds(method: str, budget: int, root: Path) -> List[int]:
    if not root.exists():
        return []
    out = []
    for d in root.iterdir():
        prefix = f"{method}_b{budget}_s"
        if d.is_dir() and d.name.startswith(prefix):
            try:
                out.append(int(d.name[len(prefix):]))
            except ValueError:
                pass
    return sorted(out)


# Aggregation
def aggregate_dataset(root: Path, dataset_label: str) -> pd.DataFrame:
    """Build per-method aggregate of accuracy + comms metrics."""
    rows = []
    for method in METHODS_HEADLINE:
        budget = 2 if method == 'abfb' else 0
        seeds = discover_seeds(method, budget, root)
        finals, peaks, bits_cum, energy_cum, last_alive_rounds = [], [], [], [], []
        for s in seeds:
            df = load_round_csv(method, budget, s, root)
            if df is None: continue
            # Final = last-round accuracy in the CSV (may be frozen if federation died)
            finals.append(df['accuracy_pct'].iloc[-1])
            peaks.append(df['accuracy_pct'].max())
            bits_cum.append(df['bits_cumulative'].iloc[-1])
            energy_cum.append(df['energy_j_cumulative'].iloc[-1])
            # Last round with at least one client alive (n_dead < N)
            n_dead_col = df['n_dead']
            N_total = df['n_dead'].max() if df['n_dead'].max() > 0 else 10
            # n_dead value indicates how many died; alive = N - n_dead
            # Last round where at least one client was still alive
            alive_mask = n_dead_col < 10
            if alive_mask.any():
                last_alive_rounds.append(int(df[alive_mask]['round'].max()))
            else:
                last_alive_rounds.append(0)
        if not finals: continue
        rows.append({
            'dataset':       dataset_label,
            'method':        method,
            'n_seeds':       len(finals),
            'final_acc_mean': np.mean(finals),
            'final_acc_std':  np.std(finals, ddof=1) if len(finals) > 1 else 0,
            'peak_acc_mean':  np.mean(peaks),
            'peak_acc_std':   np.std(peaks, ddof=1) if len(peaks) > 1 else 0,
            'bits_mean_M':    np.mean(bits_cum) / 1e6,
            'energy_mean_kJ': np.mean(energy_cum) / 1e3,
            'last_alive_mean': np.mean(last_alive_rounds),
        })
    return pd.DataFrame(rows)


# Statistical tests
def stat_tests_vs_abfb(root: Path, dataset_label: str, metric: str = 'peak_acc') -> pd.DataFrame:
    """
    Paired-t and Mann-Whitney comparisons of ABFB(B=2) vs each baseline,
    on either peak accuracy ('peak_acc') or final accuracy ('final_acc').
    """
    seeds_abfb = discover_seeds('abfb', 2, root)
    abfb_vals = []
    for s in seeds_abfb:
        df = load_round_csv('abfb', 2, s, root)
        if df is None: continue
        abfb_vals.append(df['accuracy_pct'].max() if metric == 'peak_acc'
                          else df['accuracy_pct'].iloc[-1])

    rows = []
    for method in ['fedavg', 'poc', 'oort', 'gt_linucb']:
        seeds_b = discover_seeds(method, 0, root)
        b_vals = []
        for s in seeds_b:
            df = load_round_csv(method, 0, s, root)
            if df is None: continue
            b_vals.append(df['accuracy_pct'].max() if metric == 'peak_acc'
                          else df['accuracy_pct'].iloc[-1])
        n = min(len(abfb_vals), len(b_vals))
        if n < 2:
            rows.append({'comparison': f'ABFB vs {method}', 'dataset': dataset_label,
                         'metric': metric, 'n': n, 'note': 'insufficient seeds'})
            continue
        a = np.array(abfb_vals[:n])
        b = np.array(b_vals[:n])
        # Paired-t (when seeds match across runs  -  they do, by construction)
        t_stat, t_p = stats.ttest_rel(a, b)
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative='two-sided')
        cohens_d = (np.mean(a) - np.mean(b)) / np.std(np.concatenate([a, b]), ddof=1)
        rows.append({
            'comparison':  f'ABFB vs {method}',
            'dataset':     dataset_label,
            'metric':      metric,
            'n':           n,
            'abfb_mean':   np.mean(a),
            'other_mean':  np.mean(b),
            'diff':        np.mean(a) - np.mean(b),
            't_stat':      t_stat,
            't_p':         t_p,
            'mw_u':        u_stat,
            'mw_p':        u_p,
            'cohens_d':    cohens_d,
            'note':        '',
        })
    return pd.DataFrame(rows)


# Plotting
COLORS = {
    'fedavg':    '#7f7f7f',
    'fedprox':   '#bcbd22',
    'poc':       '#17becf',
    'oort':      '#e377c2',
    'gt_linucb': '#1f77b4',
    'abfb':      '#d62728',
}
LABELS = {
    'fedavg':    'FedAvg', 'fedprox': 'FedProx', 'poc': 'Power-of-Choice',
    'oort': 'Oort', 'gt_linucb': 'GT-LinUCB', 'abfb': 'ABFB (B=2)',
}


def plot_accuracy_curves(out_path: Path):
    """Dual-panel: HAR + CIFAR-10 accuracy curves over rounds, 5-method overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, (root, label) in zip(axes, [(HAR_ROOT, 'HAR-N10'), (CIFAR_ROOT, 'CIFAR-10-N10')]):
        if not root.exists():
            ax.set_title(f"{label} (no data)")
            continue
        for method in METHODS_HEADLINE:
            budget = 2 if method == 'abfb' else 0
            seeds = discover_seeds(method, budget, root)
            if not seeds: continue
            curves = []
            for s in seeds:
                df = load_round_csv(method, budget, s, root)
                if df is None: continue
                curves.append(df['accuracy_pct'].values)
            if not curves: continue
            max_len = max(len(c) for c in curves)
            arr = np.full((len(curves), max_len), np.nan)
            for i, c in enumerate(curves):
                arr[i, :len(c)] = c
            mean = np.nanmean(arr, axis=0)
            ax.plot(np.arange(1, max_len + 1), mean,
                    color=COLORS[method], label=LABELS[method], linewidth=1.6)
        ax.set_xlabel('Round')
        ax.set_ylabel('Test accuracy (%)')
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_longevity(out_path: Path):
    """Federation-alive count vs round, CIFAR-10 only."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for method in METHODS_HEADLINE:
        budget = 2 if method == 'abfb' else 0
        seeds = discover_seeds(method, budget, CIFAR_ROOT)
        if not seeds: continue
        curves = []
        for s in seeds:
            df = load_round_csv(method, budget, s, CIFAR_ROOT)
            if df is None: continue
            alive = 10 - df['n_dead'].clip(lower=0, upper=10)
            curves.append(alive.values)
        if not curves: continue
        max_len = max(len(c) for c in curves)
        arr = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            arr[i, :len(c)] = c
        mean = np.nanmean(arr, axis=0)
        ax.plot(np.arange(1, max_len + 1), mean,
                color=COLORS[method], label=LABELS[method], linewidth=1.6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Clients alive')
    ax.set_title('Federation longevity  -  CIFAR-10-N10')
    ax.set_ylim(-0.5, 10.5)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_pareto(out_path: Path):
    """ABFB B=0, 2, 10  -  accuracy vs cumulative bits (CIFAR-10)."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    points = []
    for B in [0, 2, 10]:
        seeds = discover_seeds('abfb', B, CIFAR_ROOT)
        if not seeds: continue
        peak_accs, bits = [], []
        for s in seeds:
            df = load_round_csv('abfb', B, s, CIFAR_ROOT)
            if df is None: continue
            peak_accs.append(df['accuracy_pct'].max())
            bits.append(df['bits_cumulative'].iloc[-1] / 1e6)
        if peak_accs:
            points.append((B, np.mean(bits), np.mean(peak_accs),
                           np.std(peak_accs, ddof=1) if len(peak_accs) > 1 else 0))
    points = sorted(points, key=lambda p: p[1])
    if points:
        b_arr = [p[0] for p in points]
        x = [p[1] for p in points]
        y = [p[2] for p in points]
        yerr = [p[3] for p in points]
        ax.errorbar(x, y, yerr=yerr, marker='o', color='#d62728',
                    capsize=4, linewidth=1.8, markersize=8)
        for B, xi, yi, _ in points:
            ax.annotate(f'B={B}', xy=(xi, yi), xytext=(8, 8),
                        textcoords='offset points', fontsize=10)
    ax.set_xlabel('Cumulative communication (Mbit)')
    ax.set_ylabel('Peak accuracy (%)')
    ax.set_title('ABFB Pareto frontier  -  CIFAR-10-N10')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_belief_rmse(out_path: Path):
    """Belief RMSE per round per filter (battery/channel/compute), ABFB B=2 CIFAR-10."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    seeds = discover_seeds('abfb', 2, CIFAR_ROOT)
    curves_by_comp = {0: [], 1: [], 2: []}
    for s in seeds:
        belief_csv = CIFAR_ROOT / f"abfb_b2_s{s}" / "belief_log.csv"
        if not belief_csv.exists(): continue
        df = pd.read_csv(belief_csv)
        for i, col in enumerate(['rmse_battery', 'rmse_channel', 'rmse_compute']):
            if col in df.columns:
                curves_by_comp[i].append(df[col].values)

    titles = ['Battery', 'Channel', 'Compute']
    for ax, title, curves in zip(axes, titles, curves_by_comp.values()):
        if not curves:
            ax.set_title(f"{title} (no data)"); continue
        max_len = max(len(c) for c in curves)
        arr = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            arr[i, :len(c)] = c
        mean = np.nanmean(arr, axis=0)
        ax.plot(np.arange(1, max_len + 1), mean, color='#d62728', linewidth=1.6)
        ax.set_xlabel('Round'); ax.set_ylabel('RMSE vs ground truth')
        ax.set_title(title); ax.grid(True, linestyle='--', alpha=0.4)
    fig.suptitle('Belief-tracking RMSE  -  ABFB B=2 on CIFAR-10-N10', y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# Main
def main():
    print("=== Aggregating results ===")
    har_table   = aggregate_dataset(HAR_ROOT, 'HAR-N10') if HAR_ROOT.exists() else pd.DataFrame()
    cifar_table = aggregate_dataset(CIFAR_ROOT, 'CIFAR-10-N10')
    table = pd.concat([har_table, cifar_table], ignore_index=True)
    table.to_csv(OUT_DIR / "results_table.csv", index=False)
    print(table.to_string(index=False))

    print("\n=== Statistical tests: peak accuracy ===")
    tests_har_peak   = stat_tests_vs_abfb(HAR_ROOT, 'HAR', 'peak_acc') if HAR_ROOT.exists() else pd.DataFrame()
    tests_cifar_peak = stat_tests_vs_abfb(CIFAR_ROOT, 'CIFAR-10', 'peak_acc')
    peak_tests = pd.concat([tests_har_peak, tests_cifar_peak], ignore_index=True)
    print(peak_tests.to_string(index=False))

    print("\n=== Statistical tests: final accuracy ===")
    tests_har_final   = stat_tests_vs_abfb(HAR_ROOT, 'HAR', 'final_acc') if HAR_ROOT.exists() else pd.DataFrame()
    tests_cifar_final = stat_tests_vs_abfb(CIFAR_ROOT, 'CIFAR-10', 'final_acc')
    final_tests = pd.concat([tests_har_final, tests_cifar_final], ignore_index=True)
    print(final_tests.to_string(index=False))

    all_tests = pd.concat([peak_tests, final_tests], ignore_index=True)
    all_tests.to_csv(OUT_DIR / "stat_tests.csv", index=False)

    print("\n=== Generating figures ===")
    plot_accuracy_curves(OUT_DIR / "fig_acc.pdf")
    plot_longevity(OUT_DIR / "fig_longevity.pdf")
    plot_pareto(OUT_DIR / "fig_pareto.pdf")
    plot_belief_rmse(OUT_DIR / "fig_belief.pdf")
    # PNG mirrors for quick view
    plot_accuracy_curves(OUT_DIR / "fig_acc.png")
    plot_longevity(OUT_DIR / "fig_longevity.png")
    plot_pareto(OUT_DIR / "fig_pareto.png")
    plot_belief_rmse(OUT_DIR / "fig_belief.png")

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
