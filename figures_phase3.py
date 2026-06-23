"""Figure and statistics generator for the journal-version (extended) results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# Styling (publication-friendly, B/W-printable)
plt.rcParams.update({
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "savefig.dpi":     200,
    "savefig.bbox":    "tight",
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "lines.linewidth": 1.6,
})

METHOD_COLOR = {
    "fedavg":    "#7f7f7f",   # grey
    "fedprox":   "#bcbd22",   # olive
    "poc":       "#17becf",   # cyan
    "oort":      "#e377c2",   # pink
    "gt_linucb": "#1f77b4",   # blue
    "abfb":      "#d62728",   # red
}
METHOD_DISPLAY = {
    "fedavg":    "FedAvg",
    "fedprox":   "FedProx",
    "poc":       "Power-of-Choice",
    "oort":      "Oort",
    "gt_linucb": "GT-LinUCB",
    "abfb":      "ABFB ($B{=}2$)",
}
CONFIG_ORDER = [
    ("har",  10),
    ("har",  50),
    ("dhcd", 10),
    ("dhcd", 50),
]
CONFIG_DISPLAY = {
    ("har",  10): "HAR-N10",
    ("har",  50): "HAR-N50",
    ("dhcd", 10): "DHCD-N10",
    ("dhcd", 50): "DHCD-N50",
}


# Helpers
def safe_load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[warn] {path} not found; skipping.")
        return None
    return pd.read_csv(path)


def per_run_dir(results_root: Path, method: str, dataset: str, n: int,
                budget: int, seed: int, label: str = "") -> Path:
    base = results_root / f"phase3_results_{dataset}_n{n}"
    if label:
        return base / f"{label}_s{seed}"
    return base / f"{method}_b{budget}_s{seed}"


def load_round_log(run_dir: Path, method: str) -> Optional[pd.DataFrame]:
    """The tracker writes <system>_rounds.csv. Find any rounds CSV."""
    candidates = list(run_dir.glob("*_rounds.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


# Figure 1: accuracy curves (round vs accuracy, per (dataset,N), 6 methods)
def fig01_accuracy_curves(headline_df: pd.DataFrame, results_root: Path,
                           out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (ds, n) in zip(axes.flat, CONFIG_ORDER):
        sub = headline_df[(headline_df.dataset == ds) & (headline_df.n_clients == n)]
        for method in ["fedavg", "fedprox", "poc", "oort", "gt_linucb", "abfb"]:
            seeds_curves = []
            for seed in sorted(sub[sub.method == method].seed.unique()):
                budget = 2 if method == "abfb" else 0
                d = per_run_dir(results_root, method, ds, n, budget, seed)
                rounds_df = load_round_log(d, method)
                if rounds_df is not None and "accuracy_pct" in rounds_df.columns:
                    seeds_curves.append(rounds_df.accuracy_pct.values)
            if not seeds_curves:
                continue
            # Pad to common length
            L = max(len(c) for c in seeds_curves)
            arr = np.full((len(seeds_curves), L), np.nan)
            for i, c in enumerate(seeds_curves):
                arr[i, :len(c)] = c
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr, axis=0)
            rounds = np.arange(1, L + 1)
            ax.plot(rounds, mean, label=METHOD_DISPLAY[method],
                     color=METHOD_COLOR[method])
            ax.fill_between(rounds, mean - std, mean + std,
                             color=METHOD_COLOR[method], alpha=0.15)
        ax.set_title(CONFIG_DISPLAY[(ds, n)])
        ax.set_xlabel("Round")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(loc="lower right", ncol=2, framealpha=0.95)
    fig.suptitle("Test accuracy vs round across configurations (5-seed mean ± std)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# Figure 2: belief tracking RMSE per component
def fig02_belief_tracking(results_root: Path, out_path: Path) -> None:
    """Read belief_log.csv from ABFB runs and plot RMSE-vs-round per component."""
    components = ["rmse_battery", "rmse_channel", "rmse_compute"]
    comp_labels = ["Battery", "Channel", "Compute"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, comp, label in zip(axes, components, comp_labels):
        for ds, n in CONFIG_ORDER:
            run_dirs = list((results_root / f"phase3_results_{ds}_n{n}").glob("abfb_b2_s*"))
            curves = []
            for rd in run_dirs:
                p = rd / "belief_log.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                if comp in df.columns:
                    curves.append(df[comp].values)
            if not curves:
                continue
            L = max(len(c) for c in curves)
            arr = np.full((len(curves), L), np.nan)
            for i, c in enumerate(curves):
                arr[i, :len(c)] = c
            mean = np.nanmean(arr, axis=0)
            ax.plot(np.arange(1, L + 1), mean, label=CONFIG_DISPLAY[(ds, n)])
        ax.set_title(label)
        ax.set_xlabel("Round")
        ax.set_ylabel("RMSE vs ground truth")
        ax.legend(loc="upper right", framealpha=0.95)
        ax.set_ylim(bottom=0)
    fig.suptitle("Belief-tracking RMSE per component (mean over 5 seeds, ABFB B=2)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# Figure 3: metadata overhead  -  selection metadata bits per method
def fig03_metadata_overhead(headline_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of cumulative_bits per method × config (5-seed mean)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    methods_order = ["fedavg", "fedprox", "poc", "oort", "gt_linucb", "abfb"]
    for ax, ds in zip(axes, ["har", "dhcd"]):
        sub = headline_df[headline_df.dataset == ds]
        x = np.arange(2)   # N=10, N=50
        width = 0.13
        for i, m in enumerate(methods_order):
            vals = []
            for n in [10, 50]:
                v = sub[(sub.method == m) & (sub.n_clients == n)].cumulative_bits
                vals.append(v.mean() / 1e6 if len(v) else 0)
            ax.bar(x + (i - 2.5) * width, vals, width,
                   label=METHOD_DISPLAY[m], color=METHOD_COLOR[m])
        ax.set_xticks(x)
        ax.set_xticklabels(["N=10", "N=50"])
        ax.set_ylabel("Cumulative communication (Mbit)")
        ax.set_title(f"Dataset: {ds.upper()}")
        if ds == "har":
            ax.legend(loc="upper left", ncol=2, framealpha=0.95)
    fig.suptitle("Cumulative communication per method "
                  "(payload + selection metadata, 5-seed mean)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# Figure 4: Pareto frontier  -  accuracy vs query rate
def fig04_pareto_frontier(pareto_df: pd.DataFrame, headline_df: pd.DataFrame,
                          out_path: Path) -> None:
    if pareto_df is None:
        print("  [skip] pareto_df missing")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, (ds, n) in zip(axes.flat, CONFIG_ORDER):
        sub = pareto_df[(pareto_df.dataset == ds) & (pareto_df.n_clients == n)]
        if sub.empty:
            ax.set_title(f"{CONFIG_DISPLAY[(ds, n)]}  (no data)")
            continue
        # Group by budget; x = mean_query_rate (or budget/N), y = mean accuracy
        agg = sub.groupby("budget").agg(
            qrate_mean=("mean_query_rate", "mean"),
            acc_mean=("final_accuracy", "mean"),
            acc_std=("final_accuracy", "std"),
        ).reset_index()
        agg = agg.sort_values("budget")
        ax.errorbar(agg.qrate_mean * 100, agg.acc_mean, yerr=agg.acc_std,
                     marker="o", color=METHOD_COLOR["abfb"], label="ABFB Pareto",
                     capsize=3)
        # Annotate budget values
        for _, row in agg.iterrows():
            ax.annotate(f"B={int(row.budget)}",
                         (row.qrate_mean * 100, row.acc_mean),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
        # Reference lines: GT-LinUCB upper bound, FedAvg lower bound
        if headline_df is not None:
            h_sub = headline_df[(headline_df.dataset == ds) & (headline_df.n_clients == n)]
            for m, ls in [("gt_linucb", "--"), ("fedavg", ":")]:
                v = h_sub[h_sub.method == m].final_accuracy
                if len(v):
                    ax.axhline(v.mean(), color=METHOD_COLOR[m], linestyle=ls,
                                label=f"{METHOD_DISPLAY[m]} (mean)", alpha=0.8)
        ax.set_title(CONFIG_DISPLAY[(ds, n)])
        ax.set_xlabel("Per-round query rate (%)")
        ax.set_ylabel("Final accuracy (%)")
        ax.legend(loc="lower right", framealpha=0.95)
    fig.suptitle("ABFB Pareto front: accuracy vs query rate (5-seed mean ± std)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# Figure 5: headline boxplot
def fig05_headline_boxplot(headline_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    methods_order = ["fedavg", "fedprox", "poc", "oort", "gt_linucb", "abfb"]
    for ax, (ds, n) in zip(axes, CONFIG_ORDER):
        sub = headline_df[(headline_df.dataset == ds) & (headline_df.n_clients == n)]
        data = [sub[sub.method == m].final_accuracy.values for m in methods_order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                         labels=[METHOD_DISPLAY[m].replace(" ($B{=}2$)", "") for m in methods_order])
        for patch, m in zip(bp["boxes"], methods_order):
            patch.set_facecolor(METHOD_COLOR[m]); patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.5)
        ax.set_title(CONFIG_DISPLAY[(ds, n)])
        ax.set_ylabel("Final accuracy (%)")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Per-seed final accuracy distribution (5 seeds per cell)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# Tables
def tab_headline(headline_df: pd.DataFrame, out_path: Path) -> None:
    """Generate LaTeX table content for the headline numbers (mean ± std)."""
    methods_order = ["fedavg", "fedprox", "poc", "oort", "gt_linucb", "abfb"]
    metrics = [
        ("final_accuracy",      "Accuracy (\\%)",         "{:.2f}"),
        ("cumulative_bits",     "Communication (Mbit)",   "{:.1f}"),
        ("cumulative_energy_j", "Energy (kJ)",            "{:.1f}"),
    ]
    lines = []
    for metric, label, fmt in metrics:
        lines.append(f"\\multicolumn{{7}}{{@{{}}l@{{}}}}{{\\emph{{{label}}}}} \\\\")
        for ds, n in CONFIG_ORDER:
            row = [CONFIG_DISPLAY[(ds, n)]]
            for m in methods_order:
                v = headline_df[
                    (headline_df.dataset == ds)
                    & (headline_df.n_clients == n)
                    & (headline_df.method == m)
                ][metric]
                if len(v) == 0:
                    row.append("---")
                    continue
                mean = v.mean()
                std = v.std()
                if metric == "cumulative_bits":
                    mean /= 1e6; std /= 1e6
                if metric == "cumulative_energy_j":
                    mean /= 1e3; std /= 1e3
                row.append(f"{fmt.format(mean)}$\\pm${fmt.format(std)}")
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


def tab_belief_rmse(results_root: Path, out_path: Path) -> None:
    """Generate LaTeX table of final-round belief RMSE per component."""
    rows = []
    for ds, n in CONFIG_ORDER:
        rmses = {"bat": [], "ch": [], "cp": []}
        run_dirs = list((results_root / f"phase3_results_{ds}_n{n}").glob("abfb_b2_s*"))
        for rd in run_dirs:
            p = rd / "summary.json"
            if p.exists():
                d = json.loads(p.read_text())
                if "final_belief_rmse" in d:
                    rmses["bat"].append(d["final_belief_rmse"][0])
                    rmses["ch"].append(d["final_belief_rmse"][1])
                    rmses["cp"].append(d["final_belief_rmse"][2])
        if rmses["bat"]:
            rows.append(f"{CONFIG_DISPLAY[(ds,n)]} & "
                        f"{np.mean(rmses['bat']):.3f} & "
                        f"{np.mean(rmses['ch']):.3f} & "
                        f"{np.mean(rmses['cp']):.3f} \\\\")
    out_path.write_text("\n".join(rows))
    print(f"  wrote {out_path}")


def tab_stat_tests(headline_df: pd.DataFrame, out_path: Path) -> None:
    """Paired-t and Mann-Whitney U for ABFB vs each baseline."""
    rows = ["Comparison & paired-$t$ & $p_t$ & MW-$U$ & $p_{MW}$ \\\\", "\\midrule"]
    methods_other = ["fedavg", "fedprox", "poc", "oort", "gt_linucb"]
    for m in methods_other:
        abfb_acc = []
        m_acc = []
        for ds, n in CONFIG_ORDER:
            for seed in sorted(headline_df.seed.unique()):
                a = headline_df[(headline_df.dataset == ds)
                                & (headline_df.n_clients == n)
                                & (headline_df.method == "abfb")
                                & (headline_df.seed == seed)].final_accuracy
                b = headline_df[(headline_df.dataset == ds)
                                & (headline_df.n_clients == n)
                                & (headline_df.method == m)
                                & (headline_df.seed == seed)].final_accuracy
                if len(a) and len(b):
                    abfb_acc.append(float(a.iloc[0]))
                    m_acc.append(float(b.iloc[0]))
        if len(abfb_acc) < 2:
            continue
        t, p_t = stats.ttest_rel(abfb_acc, m_acc)
        try:
            u, p_u = stats.mannwhitneyu(abfb_acc, m_acc, alternative="two-sided")
        except Exception:
            u, p_u = float("nan"), float("nan")
        rows.append(f"ABFB vs {METHOD_DISPLAY[m]} & "
                    f"{t:.3f} & {p_t:.4f} & {u:.1f} & {p_u:.4f} \\\\")
    out_path.write_text("\n".join(rows))
    print(f"  wrote {out_path}")


def tab_ablation(ablation_df: pd.DataFrame, out_path: Path) -> None:
    if ablation_df is None or ablation_df.empty:
        print("  [skip] no ablation data")
        return
    # Expect a "run_dir" column populated by aggregate_summaries; parse label from path.
    def label_of(p):
        return Path(p).name.rsplit("_s", 1)[0]
    ablation_df = ablation_df.copy()
    ablation_df["label"] = ablation_df["run_dir"].apply(label_of)
    agg = ablation_df.groupby("label").agg(
        acc_mean=("final_accuracy", "mean"),
        acc_std=("final_accuracy", "std"),
    ).reset_index()
    # Full ABFB reference (from headline)
    rows = []
    for _, row in agg.iterrows():
        rows.append(f"{row.label.replace('_',' ')} & "
                    f"{row.acc_mean:.2f}$\\pm${row.acc_std:.2f} \\\\")
    out_path.write_text("\n".join(rows))
    print(f"  wrote {out_path}")


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path, default=Path("."))
    ap.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    headline = safe_load_csv(args.results_root / "final_summary_headline.csv")
    pareto   = safe_load_csv(args.results_root / "final_summary_pareto.csv")
    ablation = safe_load_csv(args.results_root / "final_summary_ablation.csv")

    if headline is None:
        print("[fatal] final_summary_headline.csv not found in results_root")
        sys.exit(1)

    print("\n[figures] Generating figures …")
    fig01_accuracy_curves(headline, args.results_root, args.out_dir / "fig01_accuracy_curves.pdf")
    fig02_belief_tracking(args.results_root,            args.out_dir / "fig02_belief_tracking.pdf")
    fig03_metadata_overhead(headline,                    args.out_dir / "fig03_metadata_overhead.pdf")
    fig04_pareto_frontier(pareto, headline,              args.out_dir / "fig04_pareto_frontier.pdf")
    fig05_headline_boxplot(headline,                     args.out_dir / "fig05_headline_boxplot.pdf")

    print("\n[tables] Generating LaTeX table stubs …")
    tab_headline(headline,                                args.out_dir / "tab_headline.tex")
    tab_belief_rmse(args.results_root,                    args.out_dir / "tab_belief_rmse.tex")
    tab_stat_tests(headline,                              args.out_dir / "tab_stat_tests.tex")
    tab_ablation(ablation,                                args.out_dir / "tab_ablation.tex")

    print("\n[done] All figures + tables written to", args.out_dir)


if __name__ == "__main__":
    main()
