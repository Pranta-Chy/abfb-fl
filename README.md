# ABFB: Active Belief-State Federated Bandits

Reference implementation accompanying the paper
**"Active Belief-State Federated Bandits for Energy-Aware Client Selection in IoT"**
(iCONEECT 2026 submission).

ABFB is a server-side client selection method for federated learning that
runs under realistic partial observability: the server does not get to
read each client's battery, channel, or compute state directly, only
what the FL protocol naturally exposes (update arrivals, ARQ counts,
end-to-end latency, innovation norms, bits-per-parameter). When the
server's posterior over a client becomes uncertain enough that it could
swing the selection decision, it can issue a bounded number of active
state queries per round, prioritised by a Value-of-Information (VoI)
score.

This repository contains the full simulator used to produce every number
in the paper, plus the analysis and figure-generation scripts.

## What is here

```
fl_simulation_phase3.py        ABFB simulator entry point (HAR / CIFAR-10)
fl_simulation_phase2.py        Baseline simulator (FedAvg, PoC, Oort,
                               GT-LinUCB, FedProx) + IoT wireless and
                               energy model; reused by ABFB.
skeleton_belief_models.py      Belief filters (battery / channel /
                               compute), ServerBeliefStore, and the
                               VoI-triggered active query mechanism.
cifar10_loader.py              In-memory CIFAR-10 loader with
                               Dirichlet(alpha=0.3) non-IID partitions.
cifar10_net.py                 3-conv-block CNN for CIFAR-10.

sweep_phase3.py                Sweep orchestrator (general).
run_conference_sweep_cifar10.py  Conference-specific CIFAR-10 sweep.
analyze_conference_results.py  Aggregates per-seed results -> tables and
                               figures used in the paper.
figures_phase3.py              Extended-version figure / statistics
                               generator (kept for the journal extension).
smoke_test_cifar10.py          Five-round sanity check.

requirements.txt               Python dependencies (PyTorch, numpy,
                               pandas, scipy, matplotlib).
```

## Setup

This was tested on:

- Windows 11, Python 3.10, PyTorch 2.x
- NVIDIA RTX 3050 Ti laptop GPU (4 GB VRAM)
- Intel Core i5-12500H, 16 GB RAM

```
git clone https://github.com/Pranta-Chy/abfb-fl
cd abfb-fl
python -m venv .venv
.\.venv\Scripts\activate          # PowerShell / cmd
pip install -r requirements.txt
```

## Datasets

CIFAR-10 is auto-downloaded by `torchvision` on first run. UCI HAR can
be downloaded from
<https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
and pointed at via the existing baseline loader.

## Reproducing the paper

The headline numbers in Table I and Table II come from these seeds:

- HAR-N10: `{42, 123, 456, 789, 2024}` (5 seeds)
- CIFAR-10-N10: `{0, 1, 2}` (3 seeds)

A full headline sweep for CIFAR-10:

```
python run_conference_sweep_cifar10.py
```

For the smaller HAR sweep (or to reproduce a single cell):

```
python fl_simulation_phase3.py --dataset har --method abfb --budget 2 --seed 42 --rounds 100
python fl_simulation_phase3.py --dataset har --method gt_linucb --budget 0 --seed 42 --rounds 100
```

To aggregate per-seed outputs into the table values and the figures
shipped with the paper:

```
python analyze_conference_results.py
```

This writes:

```
conference_paper_outputs/
  results_table.csv
  stat_tests.csv
  fig_acc.pdf / .png
  fig_pareto.pdf / .png
  fig_belief.pdf / .png
  fig_longevity.pdf / .png
```

A five-round smoke test, useful to verify the install works before
launching a full sweep:

```
python smoke_test_cifar10.py
```

## Configuration

The main knobs (defaults match the paper):

- `--dataset {har, cifar10}` - which task to train
- `--method {fedavg, poc, oort, gt_linucb, abfb, fedprox}` - selection
  rule on the server
- `--budget B` - per-round active query budget (only for `abfb`; 0
  disables active queries)
- `--seed S` - deterministic seed
- `--rounds T` - number of FL rounds (paper uses 100)
- `--n_clients N` - federation size (paper uses 10)
- `--K K` - clients picked per round (paper uses 4)
- `--epochs E` - local SGD epochs per round (paper uses 3)

The IoT energy model (Raspberry Pi Zero 2W class) and the 802.11g
wireless model are defined as module-level constants in
`fl_simulation_phase2.py`.

## Output layout

Each individual run writes a per-seed directory:

```
phase3_results_<dataset>_n<N>/<method>_b<B>_s<seed>/
  summary.json                # peak / final accuracy, cumulative bits,
                              # cumulative energy, lifetime, query count
  <method>_rounds.csv         # per-round accuracy, battery, bits, energy
  belief_log.csv              # per-round belief RMSE (ABFB only)
  query_log.csv               # which clients were queried each round
                              # (ABFB only)
  config.json                 # full run configuration snapshot
  figures/                    # per-run diagnostic plots
```

`analyze_conference_results.py` walks these directories and produces
the aggregate tables and figures.

## License

MIT. See `LICENSE`.

## Citation

If you use this code, please cite the iCONEECT 2026 paper. The BibTeX
entry will be added here once the proceedings DOI is assigned.
