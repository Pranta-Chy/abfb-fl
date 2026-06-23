"""Multi-baseline FL simulator with realistic IoT wireless and energy models.

Implements the IoTClient class, the EnergyModel, the wireless channel with
ARQ, the LinFA Q-agent compression controller, and the baselines (FedAvg,
PoC, Oort, GT-LinUCB, FedProx). Re-used by fl_simulation_phase3 for the
ABFB experiments.
"""
# Standard library
import copy
import io
import json
import math
import os
import random
import shutil
import time
import urllib.request
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")


# 0.  GLOBAL CONFIGURATION
SEEDS              = [42, 123, 456, 789, 2024]   # 5 seeds → robust statistics

# Dataset selector
DATASET            = "har"            # "mnist" or "har"

# Cross-device scale
NUM_CLIENTS        = 50               # 10 (small-scale) or 50 (cross-device)
NUM_ROUNDS         = 100
LOCAL_EPOCHS       = 3
BATCH_SIZE         = 32
LR                 = 0.01

# K scales proportionally with N (kept at 40% selection ratio for fair comparison)
K_SELECT           = max(4, NUM_CLIENTS * 4 // 10)

# Dataset-specific knobs
CLASSES_PER_CLIENT = 2                # MNIST Non-IID
SAMPLES_PER_CLASS  = max(50, 500 * 10 // NUM_CLIENTS)  # 500 for N=10, 100 for N=50
HAR_DIRICHLET_A    = 0.5              # Dirichlet concentration for HAR Non-IID

WARMUP_ROUNDS      = 10

# Baseline selection
# Each baseline runs through the same energy + channel model. Hybrid-RL always
# runs as the experimental condition. Comment out any baseline to skip it.
BASELINES_TO_RUN   = ["fedavg", "fedprox", "oort"]

# FedProx proximal weight (Li et al. 2020 recommend 0.001–1.0; we use 0.01)
FEDPROX_MU         = 0.01

# Oort exploration epsilon and time-penalty exponent (paper defaults)
OORT_EPSILON       = 0.1
OORT_TIME_ALPHA    = 2.0

# Edge-gateway hardware grounding
GATEWAY_VOLTAGE_V       = 3.3
GATEWAY_CPU_CURRENT_A   = 0.250
GATEWAY_TX_CURRENT_A    = 0.230
GATEWAY_RX_CURRENT_A    = 0.150
GATEWAY_IDLE_CURRENT_A  = 0.080
GATEWAY_CPU_POWER_W     = GATEWAY_VOLTAGE_V * GATEWAY_CPU_CURRENT_A   # 0.825 W
GATEWAY_TX_POWER_W      = GATEWAY_VOLTAGE_V * GATEWAY_TX_CURRENT_A    # 0.759 W
GATEWAY_RX_POWER_W      = GATEWAY_VOLTAGE_V * GATEWAY_RX_CURRENT_A    # 0.495 W
GATEWAY_IDLE_POWER_W    = GATEWAY_VOLTAGE_V * GATEWAY_IDLE_CURRENT_A  # 0.264 W

BATTERY_CAPACITY_J      = 3.7 * 2.5 * 3600    # 33,300 J (2,500 mAh × 3.7V)
WIFI_BITRATE_BPS        = 3.0e7               # 802.11g @ 30 Mbps
SECONDS_PER_MAC         = 2.0e-9              # 0.5 GFLOPS practical fp32
BACKWARD_FACTOR         = 2.5
ROUND_WALL_CLOCK_S      = 30.0
PSU_EFFICIENCY          = 0.78

QUANTIZE_OPS_PER_PARAM  = 10
INNOV_NORM_OPS_PER_PARAM = 2

# Channel / packet model
PACKET_BITS             = 1500 * 8
MAX_RETX                = 10
MIN_PACKET_PER          = 1e-5

# Reward shaping
REWARD_ACC_WEIGHT     = 100.0
REWARD_ENERGY_WEIGHT  = 0.5
REWARD_LATENCY_WEIGHT = 0.1
SKIP_PENALTY          = -2.0
ENERGY_NORM_J         = 10.0

# Q-learning hyperparameters
Q_ALPHA               = 0.3
Q_GAMMA               = 0.9
Q_EPSILON_START       = 0.3
Q_EPSILON_MIN         = 0.05
Q_EPSILON_DECAY       = 0.995

# LinUCB
LINUCB_ALPHA          = 1.0
CONTEXT_DIM           = 4

# DP
USE_DP                = False
DP_CLIP_NORM          = 1.0
DP_SIGMA              = 0.5

INNOV_WEIGHT_NORM     = 0.8
BIT_ACTIONS           = {0: None, 1: 4, 2: 8, 3: 16}

# Q-agent variant
Q_AGENT_TYPE          = "linfa"       # "tabular" | "linfa" | "dqn" (recommended: linfa)

# Output directory
OUT_DIR = Path(f"phase2_results_{DATASET}_n{NUM_CLIENTS}_{Q_AGENT_TYPE}")
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1.  REPRODUCIBILITY UTILITIES
def set_global_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def snapshot_rng_state():
    return (random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng_state(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def save_config(seed, baseline_name, out_path):
    cfg = {
        "phase": "phase2_q1_extensions",
        "seed": seed, "baseline": baseline_name,
        "dataset": DATASET, "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS, "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE, "lr": LR, "k_select": K_SELECT,
        "classes_per_client": CLASSES_PER_CLIENT, "samples_per_class": SAMPLES_PER_CLASS,
        "har_dirichlet_alpha": HAR_DIRICHLET_A,
        "warmup_rounds": WARMUP_ROUNDS, "q_agent_type": Q_AGENT_TYPE,
        "fedprox_mu": FEDPROX_MU, "oort_epsilon": OORT_EPSILON,
        "oort_time_alpha": OORT_TIME_ALPHA,
        "gateway_cpu_power_w": GATEWAY_CPU_POWER_W,
        "gateway_tx_power_w": GATEWAY_TX_POWER_W,
        "gateway_rx_power_w": GATEWAY_RX_POWER_W,
        "gateway_idle_power_w": GATEWAY_IDLE_POWER_W,
        "battery_capacity_j": BATTERY_CAPACITY_J,
        "wifi_bitrate_bps": WIFI_BITRATE_BPS,
        "seconds_per_mac": SECONDS_PER_MAC,
        "backward_factor": BACKWARD_FACTOR,
        "psu_efficiency": PSU_EFFICIENCY,
        "round_wall_clock_s": ROUND_WALL_CLOCK_S,
        "packet_bits": PACKET_BITS, "max_retx": MAX_RETX,
        "reward_acc_weight": REWARD_ACC_WEIGHT,
        "reward_energy_weight": REWARD_ENERGY_WEIGHT,
        "reward_latency_weight": REWARD_LATENCY_WEIGHT,
        "skip_penalty": SKIP_PENALTY,
        "q_alpha": Q_ALPHA, "q_gamma": Q_GAMMA,
        "linucb_alpha": LINUCB_ALPHA,
        "innov_weight_norm": INNOV_WEIGHT_NORM,
        "use_dp": USE_DP,
    }
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)


# 2.  WIRELESS CHANNEL MODEL (Empirical 802.11g PER + ARQ)
class WirelessChannel:
    """Indoor 802.11g PER curve (linear interpolation of empirical anchors)."""

    @staticmethod
    def base_per(rssi_dbm):
        if rssi_dbm >= -50: return 0.0005
        if rssi_dbm >= -65: return 0.0005 + (-50 - rssi_dbm) / 15 * (0.005 - 0.0005)
        if rssi_dbm >= -80: return 0.005 + (-65 - rssi_dbm) / 15 * (0.04 - 0.005)
        if rssi_dbm >= -90: return 0.04 + (-80 - rssi_dbm) / 10 * (0.30 - 0.04)
        return 0.5

    @classmethod
    def packet_error_rate(cls, rssi_dbm, n_bits=PACKET_BITS):
        base = cls.base_per(rssi_dbm)
        fluct = float(np.random.lognormal(0, 0.4))
        return float(np.clip(base * fluct, MIN_PACKET_PER, 0.5))

    @classmethod
    def transmit(cls, total_bits, rssi_dbm):
        n_packets = max(1, math.ceil(total_bits / PACKET_BITS))
        bits_on_air = 0; n_retx = 0; success = True
        for _ in range(n_packets):
            attempts = 0; delivered = False
            while attempts <= MAX_RETX:
                per = cls.packet_error_rate(rssi_dbm, PACKET_BITS)
                bits_on_air += PACKET_BITS
                if random.random() > per:
                    delivered = True; break
                attempts += 1; n_retx += 1
            if not delivered:
                success = False; break
        return success, bits_on_air, n_retx


# 3.  ENERGY MODEL (Six components: compute + Tx + Rx + idle + aux + RL)
class EnergyModel:
    @staticmethod
    def compute_energy_j(macs):
        return GATEWAY_CPU_POWER_W * (macs * SECONDS_PER_MAC)

    @staticmethod
    def tx_energy_j(bits):
        return GATEWAY_TX_POWER_W * (bits / WIFI_BITRATE_BPS)

    @staticmethod
    def rx_energy_j(bits):
        return GATEWAY_RX_POWER_W * (bits / WIFI_BITRATE_BPS)

    @staticmethod
    def idle_energy_j(seconds):
        return GATEWAY_IDLE_POWER_W * seconds

    @staticmethod
    def joules_to_battery_pct(j):
        return j / BATTERY_CAPACITY_J * 100.0


# 4.  MODELS (MNIST CNN + HAR MLP)
class MNISTNet(nn.Module):
    """Lightweight CNN  -  21,306 parameters."""
    FORWARD_MACS_PER_SAMPLE = 1_218_048
    TOTAL_MACS_PER_STEP     = int(FORWARD_MACS_PER_SAMPLE * (1 + BACKWARD_FACTOR))
    N_CLASSES               = 10
    INPUT_SHAPE             = (1, 28, 28)

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class HARNet(nn.Module):
    """Lightweight MLP for HAR  -  561-dim feature input → 6 activity classes.

    Why MLP not CNN for HAR: UCI HAR features are already pre-extracted (mean,
    std, FFT components, etc., from raw IMU data). The 561-feature vector is
    a "wide-flat" representation; a 3-layer MLP is the canonical baseline
    that matches the dataset's structure. Total parameters: ~80,710.
    """
    # Forward MACs per sample = 561*128 + 128*64 + 64*6 = 79,872
    FORWARD_MACS_PER_SAMPLE = 79_872
    TOTAL_MACS_PER_STEP     = int(FORWARD_MACS_PER_SAMPLE * (1 + BACKWARD_FACTOR))
    N_CLASSES               = 6
    INPUT_SHAPE             = (561,)

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(561, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def get_model_class():
    return MNISTNet if DATASET == "mnist" else HARNet


def count_param_bits(model, bits=32):
    return sum(p.numel() for p in model.parameters()) * bits


def estimate_compute_macs(n_samples, epochs, batch_size, model_cls=None):
    if model_cls is None:
        model_cls = get_model_class()
    n_steps = epochs * math.ceil(n_samples / batch_size)
    return n_steps * batch_size * model_cls.TOTAL_MACS_PER_STEP


# 5.  DATASETS

# 5a. MNIST Non-IID
def build_mnist_loaders(num_clients, classes_per_client, batch_size,
                        samples_per_class, seed):
    """Non-IID MNIST: each client gets `classes_per_client` classes cyclically."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform)

    rng = random.Random(seed)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(train_ds):
        class_indices[label].append(idx)
    for k in class_indices:
        rng.shuffle(class_indices[k])

    client_loaders = []
    n_samples_per_client = []

    for c in range(num_clients):
        assigned = [(c * classes_per_client + k) % 10
                    for k in range(classes_per_client)]
        indices = []
        for cls in assigned:
            cls_idx = class_indices[cls]
            start = (c * samples_per_class) % len(cls_idx)
            end = start + samples_per_class
            if end <= len(cls_idx):
                indices.extend(cls_idx[start:end])
            else:
                indices.extend(cls_idx[start:])
                indices.extend(cls_idx[:end - len(cls_idx)])

        loader = DataLoader(Subset(train_ds, indices), batch_size=batch_size,
                            shuffle=True, num_workers=0, pin_memory=False)
        client_loaders.append(loader)
        n_samples_per_client.append(len(indices))

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=0, pin_memory=False)
    return client_loaders, test_loader, n_samples_per_client


# 5b. UCI HAR dataset loader
HAR_URLS = [
    # Primary UCI mirror
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    # Backup mirror (may exist)
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]


def _download_har_dataset(target_dir):
    """Download UCI HAR dataset and unpack to target_dir."""
    target_dir = Path(target_dir)
    marker = target_dir / "X_train.txt"
    if marker.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "har.zip"

    last_err = None
    for url in HAR_URLS:
        try:
            print(f"[HAR] Downloading from {url} ...")
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as e:
            last_err = e
            print(f"[HAR] Download failed: {e}; trying next mirror")
    if not zip_path.exists():
        raise RuntimeError(
            f"Failed to download HAR dataset from all mirrors. Last error: {last_err}\n"
            f"Manual fix: download the UCI HAR Dataset zip and extract X_train.txt, "
            f"y_train.txt, subject_train.txt (and test/ counterparts) into {target_dir}"
        )

    print(f"[HAR] Extracting to {target_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(target_dir)

    # The UCI zip has a nested "UCI HAR Dataset" directory. Flatten it.
    nested = target_dir / "UCI HAR Dataset"
    if nested.exists():
        for split in ["train", "test"]:
            for stem in [f"X_{split}", f"y_{split}", f"subject_{split}"]:
                src = nested / split / f"{stem}.txt"
                dst = target_dir / f"{stem}.txt"
                if src.exists() and not dst.exists():
                    shutil.copyfile(src, dst)
        # Also copy features.txt and activity_labels.txt
        for stem in ["features", "activity_labels"]:
            src = nested / f"{stem}.txt"
            dst = target_dir / f"{stem}.txt"
            if src.exists() and not dst.exists():
                shutil.copyfile(src, dst)

    zip_path.unlink(missing_ok=True)
    if not marker.exists():
        raise RuntimeError(
            f"HAR extraction did not produce X_train.txt in {target_dir}; "
            f"please verify the archive structure manually."
        )
    return target_dir


class HARDataset(Dataset):
    """UCI Human Activity Recognition (561 features, 6 classes, 30 subjects).

    Reference: Anguita et al. (2013), "A Public Domain Dataset for Human
    Activity Recognition Using Smartphones." ESANN.
    Classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
    """

    def __init__(self, root="./data/har", train=True):
        data_dir = _download_har_dataset(root)
        split = "train" if train else "test"
        self.X = np.loadtxt(data_dir / f"X_{split}.txt", dtype=np.float32)
        self.y = np.loadtxt(data_dir / f"y_{split}.txt", dtype=np.int64) - 1  # 1-indexed → 0-indexed
        self.subjects = np.loadtxt(data_dir / f"subject_{split}.txt", dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


def build_har_loaders(num_clients, batch_size, seed,
                     dirichlet_alpha=HAR_DIRICHLET_A):
    """Partition HAR for federated learning.

    Strategy:
      - If num_clients == 21: 1:1 subject-to-client (natural Non-IID).
      - Otherwise: Dirichlet-α partition of all training samples across clients.
        This is the canonical FL Non-IID partition (Hsu et al. 2019). Lower α
        means more heterogeneous; α = 0.5 is moderate Non-IID.
    """
    train_ds = HARDataset(train=True)
    test_ds = HARDataset(train=False)

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    train_subjects = np.unique(train_ds.subjects)  # 21 subjects in train split

    client_indices = [[] for _ in range(num_clients)]

    if num_clients == len(train_subjects):
        # 1:1 subject → client (natural cross-subject Non-IID)
        shuffled = list(train_subjects); rng.shuffle(shuffled)
        for c, subj in enumerate(shuffled):
            client_indices[c] = list(np.where(train_ds.subjects == subj)[0])
    else:
        # Dirichlet partition per class
        n_classes = HARNet.N_CLASSES
        for cls in range(n_classes):
            cls_idx = np.where(train_ds.y == cls)[0]
            np_rng.shuffle(cls_idx)
            # Dirichlet proportions across clients
            props = np_rng.dirichlet(np.ones(num_clients) * dirichlet_alpha)
            # Convert to counts
            counts = (props * len(cls_idx)).astype(int)
            # Fix rounding error
            counts[-1] = len(cls_idx) - counts[:-1].sum()
            start = 0
            for c, n in enumerate(counts):
                client_indices[c].extend(cls_idx[start:start + n].tolist())
                start += n

    client_loaders = []
    n_samples_per_client = []
    for c in range(num_clients):
        idx = client_indices[c]
        if len(idx) == 0:
            # Empty client  -  give it one sample to avoid loader errors
            idx = [0]
        loader = DataLoader(Subset(train_ds, idx), batch_size=batch_size,
                            shuffle=True, num_workers=0, pin_memory=False)
        client_loaders.append(loader)
        n_samples_per_client.append(len(idx))

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=0, pin_memory=False)
    return client_loaders, test_loader, n_samples_per_client


def get_dataset_loaders(seed):
    if DATASET == "mnist":
        return build_mnist_loaders(NUM_CLIENTS, CLASSES_PER_CLIENT,
                                    BATCH_SIZE, SAMPLES_PER_CLASS, seed)
    elif DATASET == "har":
        return build_har_loaders(NUM_CLIENTS, BATCH_SIZE, seed)
    raise ValueError(f"Unknown DATASET: {DATASET}")


# 6.  DELTA QUANTIZATION
class DeltaQuantizer:
    @staticmethod
    def quantize_delta(local_model, global_model, bits):
        if bits == 32:
            return {k: (lp.data - gp.data).clone()
                    for (k, lp), (_, gp) in zip(
                        local_model.named_parameters(),
                        global_model.named_parameters())}
        levels = 2 ** bits
        out = {}
        with torch.no_grad():
            for (k, lp), (_, gp) in zip(local_model.named_parameters(),
                                        global_model.named_parameters()):
                delta = (lp.data - gp.data).float()
                v_min, v_max = delta.min(), delta.max()
                if v_max == v_min:
                    out[k] = torch.zeros_like(delta); continue
                scale = (v_max - v_min) / (levels - 1)
                q_int = torch.clamp(torch.round((delta - v_min) / scale),
                                    0, levels - 1)
                out[k] = (q_int * scale + v_min).to(delta.dtype)
        return out


# 7.  DIFFERENTIAL PRIVACY (optional Gaussian mechanism)
def apply_dp_to_delta(delta_dict, clip_norm=DP_CLIP_NORM, sigma=DP_SIGMA):
    total = math.sqrt(sum(d.pow(2).sum().item() for d in delta_dict.values()))
    factor = min(1.0, clip_norm / (total + 1e-12))
    return {k: (d * factor + torch.randn_like(d) * (sigma * clip_norm))
            for k, d in delta_dict.items()}


# 8.  METRICS TRACKER (per-baseline aware)
class MetricsTracker:
    def __init__(self, system_name):
        self.system_name = system_name      # "Hybrid-RL", "FedAvg", "FedProx", "Oort"
        self._bits_cum = 0
        self._energy_cum = 0.0
        self.rounds = []                    # one dict per round
        self.bits_log = []
        self.energy_log = []
        self.client_history = []
        self.energy_breakdown = []
        self.retx_log = []

    def log_round(self, rnd, accuracy, avg_battery, bits_round,
                  energy_round, e_compute, e_tx, e_rx, e_idle,
                  e_aux, e_rl, selected_ids, n_skipped, n_dead, n_retx,
                  avg_innov=0.0, reward_mean=0.0):
        self._bits_cum   += bits_round
        self._energy_cum += energy_round
        self.bits_log.append(self._bits_cum)
        self.energy_log.append(self._energy_cum)
        self.rounds.append({
            "round": rnd, "accuracy_pct": round(accuracy*100, 4),
            "avg_battery_pct": round(avg_battery, 4),
            "bits_this_round": bits_round,
            "bits_cumulative": self._bits_cum,
            "energy_j_this_round": round(energy_round, 6),
            "energy_j_cumulative": round(self._energy_cum, 6),
            "e_compute_j": round(e_compute, 6),
            "e_tx_j":      round(e_tx, 6),
            "e_rx_j":      round(e_rx, 6),
            "e_idle_j":    round(e_idle, 6),
            "e_aux_j":     round(e_aux, 6),
            "e_rl_j":      round(e_rl, 9),
            "selected_ids": str(selected_ids),
            "n_skipped": n_skipped, "n_dead": n_dead, "n_retx": n_retx,
            "avg_innovation": round(avg_innov, 4),
            "mean_reward": round(reward_mean, 4),
        })
        self.energy_breakdown.append({
            "round": rnd, "system": self.system_name,
            "compute_j": e_compute, "tx_j": e_tx, "rx_j": e_rx,
            "idle_j": e_idle, "aux_j": e_aux, "rl_j": e_rl,
        })
        self.retx_log.append({"round": rnd, "system": self.system_name,
                              "n_retx": n_retx})

    def log_client_state(self, rnd, cid, battery, rssi, selected,
                         action, bits_sent, energy_j):
        self.client_history.append({
            "round": rnd, "system": self.system_name, "client_id": cid,
            "battery_pct": round(battery, 4),
            "rssi_dbm":   round(rssi, 4),
            "selected": selected, "action": action,
            "bits_sent": bits_sent, "energy_j": round(energy_j, 5),
        })

    def save_all(self, out_dir):
        prefix = self.system_name.lower().replace("-", "_")
        pd.DataFrame(self.rounds).to_csv(out_dir / f"{prefix}_rounds.csv", index=False)


# 9.  IoT CLIENT  (with FedProx-aware local_train)
class IoTClient:
    def __init__(self, client_id, data_loader, n_samples,
                 battery_pct=None, rssi=None, compute_factor=None):
        self.id = client_id
        self.data_loader = data_loader
        self.n_samples = n_samples
        init_pct = battery_pct if battery_pct is not None else random.uniform(60, 100)
        self.battery_j = (init_pct / 100.0) * BATTERY_CAPACITY_J
        self.rssi = rssi if rssi is not None else random.uniform(-65, -45)
        self.compute_factor = (compute_factor if compute_factor is not None
                                else random.uniform(0.5, 1.5))
        self.staleness = 0
        self.q_agent = None              # set later by Hybrid-RL run
        self.is_dead = False
        self.last_local_loss = 0.0       # tracked for Oort utility

    @property
    def battery_pct(self):
        return EnergyModel.joules_to_battery_pct(self.battery_j)

    def step_environment(self):
        self.rssi = float(np.clip(self.rssi + random.uniform(-2, 2), -85, -35))

    def drain_j(self, joules):
        actual = joules / PSU_EFFICIENCY
        self.battery_j = max(0.0, self.battery_j - actual)
        if self.battery_j == 0.0:
            self.is_dead = True

    def credit_j(self, joules):
        actual = joules / PSU_EFFICIENCY
        self.battery_j = min(BATTERY_CAPACITY_J, self.battery_j + actual)

    def drain_idle(self):
        self.drain_j(EnergyModel.idle_energy_j(ROUND_WALL_CLOCK_S))

    def refund_idle_for_active_period(self, active_time_s):
        active_time_s = min(active_time_s, ROUND_WALL_CLOCK_S)
        self.credit_j(GATEWAY_IDLE_POWER_W * active_time_s)

    def drain_rx(self, bits):
        e_j = EnergyModel.rx_energy_j(bits)
        self.drain_j(e_j)
        return e_j

    def drain_q_inference(self):
        if self.q_agent is None: return 0.0
        macs = self.q_agent.INFERENCE_MACS * self.compute_factor
        e_j = EnergyModel.compute_energy_j(macs)
        self.drain_j(e_j); return e_j

    def drain_q_update(self):
        if self.q_agent is None: return 0.0
        macs = self.q_agent.UPDATE_MACS * self.compute_factor
        e_j = EnergyModel.compute_energy_j(macs)
        self.drain_j(e_j); return e_j

    def drain_quantize_compute(self, n_params):
        macs = n_params * QUANTIZE_OPS_PER_PARAM * self.compute_factor
        e_j = EnergyModel.compute_energy_j(macs)
        self.drain_j(e_j); return e_j

    def drain_innov_norm_compute(self, n_params):
        macs = n_params * INNOV_NORM_OPS_PER_PARAM * self.compute_factor
        e_j = EnergyModel.compute_energy_j(macs)
        self.drain_j(e_j); return e_j

    def estimated_local_train_energy_j(self):
        model_cls = get_model_class()
        macs = (estimate_compute_macs(self.n_samples, LOCAL_EPOCHS, BATCH_SIZE, model_cls)
                * self.compute_factor)
        return EnergyModel.compute_energy_j(macs)

    def active_time_for_compute(self, joules):
        return joules / GATEWAY_CPU_POWER_W

    def active_time_for_tx(self, bits):
        return bits / WIFI_BITRATE_BPS

    def active_time_for_rx(self, bits):
        return bits / WIFI_BITRATE_BPS

    def local_train(self, global_model, fedprox_mu=0.0):
        """Local SGD training. Drains compute energy and tracks last_local_loss.

        If fedprox_mu > 0, applies the FedProx proximal term:
            loss += (mu/2) * ||w_local - w_global||²
        """
        e = self.estimated_local_train_energy_j()
        self.drain_j(e)

        local_model = copy.deepcopy(global_model).to(DEVICE)
        global_params = None
        if fedprox_mu > 0:
            global_params = [p.detach().clone() for p in global_model.parameters()]

        opt = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        local_model.train()

        running_loss = 0.0; n_batches = 0
        for _ in range(LOCAL_EPOCHS):
            for X, y in self.data_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logits = local_model(X)
                loss = criterion(logits, y)
                if global_params is not None:
                    prox = sum(((p - gp) ** 2).sum()
                               for p, gp in zip(local_model.parameters(), global_params))
                    loss = loss + 0.5 * fedprox_mu * prox
                loss.backward()
                opt.step()
                running_loss += float(loss.detach().cpu()); n_batches += 1

        self.last_local_loss = running_loss / max(1, n_batches)
        return local_model

    def get_context(self):
        return np.array([
            self.battery_pct / 100.0,
            (self.rssi + 90) / 60.0,
            min(self.staleness, 10) / 10.0,
            (self.compute_factor - 0.5) / 1.0,
        ], dtype=np.float64)


# 10.  LinUCB (alive-only filtering)
class LinUCBAgent:
    def __init__(self, n_arms, context_dim=CONTEXT_DIM, alpha=LINUCB_ALPHA):
        self.n_arms, self.d, self.alpha = n_arms, context_dim, alpha
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.selection_counts = np.zeros(n_arms, dtype=int)

    def select_clients(self, contexts, k):
        if not contexts: return []
        scores = {}
        for arm, x in contexts.items():
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            ucb   = self.alpha * np.sqrt(x @ A_inv @ x)
            scores[arm] = float(theta @ x) + ucb
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [arm for arm, _ in ranked[:k]]
        for s in selected:
            self.selection_counts[s] += 1
        return selected

    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

    def jain_fairness(self):
        c = self.selection_counts.astype(float)
        if c.sum() == 0: return 0.0
        return float((c.sum() ** 2) / (len(c) * (c ** 2).sum()))


# 11.  Q-AGENT VARIANTS  (Tabular | LinFA | DQN)
class _QAgentBase:
    INFERENCE_MACS = 0
    UPDATE_MACS    = 0
    def __init__(self, cid):
        self.id = cid
        self.epsilon = Q_EPSILON_START
        self.gamma = Q_GAMMA
        self.action_counts = np.zeros(4, dtype=int)
        self.n_updates = 0
    def decay(self):
        self.epsilon = max(Q_EPSILON_MIN, self.epsilon * Q_EPSILON_DECAY)
    def get_action_distribution(self):
        return {("skip" if BIT_ACTIONS[a] is None else f"{BIT_ACTIONS[a]}-bit"):
                int(self.action_counts[a]) for a in range(4)}


class TabularQAgent(_QAgentBase):
    BATTERY_BOUNDS    = (30, 70)
    RSSI_BOUNDS       = (-70, -50)
    INNOVATION_BOUNDS = (0.4, 0.8)
    INFERENCE_MACS    = 5
    UPDATE_MACS       = 10

    def __init__(self, cid):
        super().__init__(cid)
        self.alpha = Q_ALPHA
        self.q_table = np.zeros((3, 3, 3, 4), dtype=np.float64)
        self.q_table[:, :, :, 0] = -1.0

    @staticmethod
    def _bucket(v, lo, hi):
        return 0 if v < lo else (1 if v < hi else 2)

    def get_state(self, battery_pct, rssi, innov_norm, compute_factor=1.0):
        return (self._bucket(battery_pct, *self.BATTERY_BOUNDS),
                self._bucket(rssi, *self.RSSI_BOUNDS),
                self._bucket(innov_norm, *self.INNOVATION_BOUNDS))

    def select_action(self, state, allow_skip=True):
        if random.random() < self.epsilon:
            a = random.randint(0, 3) if allow_skip else random.randint(1, 3)
        else:
            a = int(np.argmax(self.q_table[state])) if allow_skip \
                else int(np.argmax(self.q_table[state][1:])) + 1
        self.action_counts[a] += 1
        return a

    def update(self, s, a, r, sn):
        td_target = r + self.gamma * np.max(self.q_table[sn])
        self.q_table[s][a] += self.alpha * (td_target - self.q_table[s][a])
        self.n_updates += 1

    def policy_table(self):
        return np.argmax(self.q_table, axis=-1)


class LinFAQAgent(_QAgentBase):
    FEAT_DIM = 5
    INFERENCE_MACS = 20
    UPDATE_MACS    = 40

    def __init__(self, cid):
        super().__init__(cid)
        self.alpha = 0.05
        self.theta = np.zeros((4, self.FEAT_DIM), dtype=np.float64)
        self.theta[0, -1] = -1.0

    @staticmethod
    def _features(b, r, i, c):
        return np.array([b/100.0, (r+90)/60.0, min(i,5.0)/5.0,
                         (c-0.5)/1.0, 1.0], dtype=np.float64)

    def get_state(self, b, r, i, c=1.0):
        return self._features(b, r, i, c)

    def _q_values(self, phi, allow_skip=True):
        q = self.theta @ phi
        if not allow_skip: q = q.copy(); q[0] = -np.inf
        return q

    def select_action(self, state, allow_skip=True):
        if random.random() < self.epsilon:
            a = random.randint(0, 3) if allow_skip else random.randint(1, 3)
        else:
            a = int(np.argmax(self._q_values(state, allow_skip)))
        self.action_counts[a] += 1
        return a

    def update(self, s, a, r, sn):
        td = r + self.gamma * float(np.max(self.theta @ sn)) - float(self.theta[a] @ s)
        self.theta[a] += self.alpha * td * s
        self.theta[a] *= (1.0 - 1e-4)
        self.n_updates += 1

    def policy_table(self):
        bat_grid=[15,50,85]; rssi_grid=[-80,-60,-40]; innov_grid=[0.2,0.6,1.0]
        out = np.zeros((3,3,3), dtype=int)
        for i,b in enumerate(bat_grid):
            for j,r in enumerate(rssi_grid):
                for k,iv in enumerate(innov_grid):
                    phi = self._features(b,r,iv,1.0)
                    out[i,j,k] = int(np.argmax(self.theta @ phi))
        return out


class TinyDQNAgent(_QAgentBase):
    FEAT_DIM = 5; HIDDEN = 16; REPLAY_MAX = 64; BATCH = 16
    TARGET_UPDATE_EVERY = 10; LR = 1e-3
    INFERENCE_MACS = 400
    UPDATE_MACS    = 27_000

    def __init__(self, cid):
        super().__init__(cid)
        self._dev = torch.device("cpu")
        self.q_net = nn.Sequential(
            nn.Linear(self.FEAT_DIM, self.HIDDEN), nn.ReLU(),
            nn.Linear(self.HIDDEN, self.HIDDEN), nn.ReLU(),
            nn.Linear(self.HIDDEN, 4),
        ).to(self._dev)
        with torch.no_grad():
            self.q_net[-1].bias.data[0] = -1.0
        self.target_net = copy.deepcopy(self.q_net).to(self._dev)
        self.target_net.eval()
        self.opt = optim.Adam(self.q_net.parameters(), lr=self.LR)
        self.replay = []

    @staticmethod
    def _features(b, r, i, c):
        return np.array([b/100.0, (r+90)/60.0, min(i,5.0)/5.0,
                         (c-0.5)/1.0, 1.0], dtype=np.float32)

    def get_state(self, b, r, i, c=1.0):
        return self._features(b, r, i, c)

    @torch.no_grad()
    def _q_values(self, phi, allow_skip=True):
        s = torch.tensor(phi, dtype=torch.float32, device=self._dev).unsqueeze(0)
        q = self.q_net(s).squeeze(0).cpu().numpy()
        if not allow_skip: q = q.copy(); q[0] = -np.inf
        return q

    def select_action(self, state, allow_skip=True):
        if random.random() < self.epsilon:
            a = random.randint(0, 3) if allow_skip else random.randint(1, 3)
        else:
            a = int(np.argmax(self._q_values(state, allow_skip)))
        self.action_counts[a] += 1
        return a

    def update(self, s, a, r, sn):
        self.replay.append((s, a, float(r), sn))
        if len(self.replay) > self.REPLAY_MAX:
            self.replay.pop(0)
        bs = min(self.BATCH, len(self.replay))
        idxs = random.sample(range(len(self.replay)), bs)
        batch = [self.replay[i] for i in idxs]
        states  = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self._dev)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self._dev)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self._dev)
        next_s  = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self._dev)
        with torch.no_grad():
            q_next = self.target_net(next_s).max(dim=1)[0]
            td_target = rewards + self.gamma * q_next
        q_curr = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_curr, td_target)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.opt.step()
        self.n_updates += 1
        if self.n_updates % self.TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def policy_table(self):
        bat_grid=[15,50,85]; rssi_grid=[-80,-60,-40]; innov_grid=[0.2,0.6,1.0]
        out = np.zeros((3,3,3), dtype=int)
        for i,b in enumerate(bat_grid):
            for j,r in enumerate(rssi_grid):
                for k,iv in enumerate(innov_grid):
                    phi = self._features(b,r,iv,1.0)
                    out[i,j,k] = int(np.argmax(self._q_values(phi, True)))
        return out


def make_q_agent(cid):
    if Q_AGENT_TYPE == "tabular": return TabularQAgent(cid)
    if Q_AGENT_TYPE == "linfa":   return LinFAQAgent(cid)
    if Q_AGENT_TYPE == "dqn":     return TinyDQNAgent(cid)
    raise ValueError(f"Unknown Q_AGENT_TYPE: {Q_AGENT_TYPE!r}")


# 12.  AGGREGATION  (BN-safe, delta-mode)
def aggregate_deltas(global_model, deltas, weights):
    if not deltas: return global_model
    total = sum(weights); weights = [w/total for w in weights]
    with torch.no_grad():
        for name, p in global_model.named_parameters():
            agg = torch.zeros_like(p.data, dtype=torch.float32)
            for d, w in zip(deltas, weights):
                agg += w * d[name].to(agg.dtype).to(p.device)
            p.data.add_(agg.to(p.data.dtype))
    return global_model


def fedavg_full_aggregate(global_model, local_models, weights=None):
    if not local_models: return global_model
    if weights is None:
        weights = [1.0 / len(local_models)] * len(local_models)
    else:
        s = sum(weights); weights = [w/s for w in weights]

    g_state = global_model.state_dict()
    new_state = {}
    for key, val in g_state.items():
        if val.is_floating_point():
            agg = torch.zeros_like(val, dtype=torch.float32)
            for m, w in zip(local_models, weights):
                agg += w * m.state_dict()[key].float()
            new_state[key] = agg.to(val.dtype)
        else:
            new_state[key] = local_models[0].state_dict()[key].clone()
    global_model.load_state_dict(new_state)
    return global_model


# 13.  EVALUATION
@torch.no_grad()
def evaluate(model, test_loader):
    model.eval(); model.to(DEVICE)
    n_classes = get_model_class().N_CLASSES
    correct = total = 0
    cm = np.zeros((n_classes, n_classes), dtype=int)
    per_class_c = np.zeros(n_classes); per_class_t = np.zeros(n_classes)
    for X, y in test_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X).argmax(dim=1)
        correct += (pred == y).sum().item(); total += y.size(0)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
            per_class_t[t] += 1
            if t == p: per_class_c[t] += 1
    per_class = np.divide(per_class_c, np.maximum(per_class_t, 1))
    return correct / total, per_class, cm


# 14.  REWARD FUNCTION  (Hybrid-RL)
def compute_reward(bits_used, acc_delta, compute_j, tx_j, tx_latency_s):
    if bits_used == 0:
        return SKIP_PENALTY - REWARD_ENERGY_WEIGHT * (compute_j / ENERGY_NORM_J)
    e_term = (compute_j + tx_j) / ENERGY_NORM_J
    return (REWARD_ACC_WEIGHT * acc_delta
            - REWARD_ENERGY_WEIGHT * e_term
            - REWARD_LATENCY_WEIGHT * tx_latency_s)


# 15.  HYBRID-RL SIMULATION  (LinUCB + Q-controller + delta-quantised aggregation)
def run_hybrid_rl(clients, test_loader, tracker, verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print(f"  SIMULATION  -  Hybrid-RL  ({Q_AGENT_TYPE})  -  {DATASET} / N={NUM_CLIENTS}")
        print("=" * 70)

    # Assign Q-agents to all clients
    for c in clients:
        c.q_agent = make_q_agent(c.id)

    model_cls = get_model_class()
    global_model = model_cls().to(DEVICE)
    linucb = LinUCBAgent(NUM_CLIENTS)
    accuracy_log = []; battery_log = []
    per_class_log = []; cm_final = None
    reward_log = []
    prev_acc = 0.0

    for rnd in range(1, NUM_ROUNDS + 1):
        is_warmup = rnd <= WARMUP_ROUNDS
        t0 = time.time()

        for c in clients: c.step_environment()
        for c in clients:
            if not c.is_dead: c.drain_idle()

        alive_contexts = {c.id: c.get_context() for c in clients if not c.is_dead}
        selected_ids = linucb.select_clients(alive_contexts, K_SELECT)

        local_deltas = []; weights = []; update_records = []
        round_bits_air = 0
        e_compute_round = 0.0; e_tx_round = 0.0; e_rx_round = 0.0
        e_aux_round = 0.0; e_rl_round = 0.0
        n_skipped = 0; n_retx_round = 0; innov_acc = []

        n_params = sum(p.numel() for p in global_model.parameters())
        downlink_bits = count_param_bits(global_model, 32)

        for cid in selected_ids:
            c = clients[cid]

            if c.battery_pct < 5.0 and not is_warmup:
                c.staleness += 1; n_skipped += 1
                state = c.q_agent.get_state(c.battery_pct, c.rssi, 0.0, c.compute_factor)
                update_records.append({
                    "cid": cid, "state": state, "action": 0,
                    "bits_air": 0, "compute_j": 0.0, "tx_j": 0.0,
                    "rx_j": 0.0, "aux_j": 0.0, "tx_latency": 0.0,
                    "active_time_s": 0.0, "innov_norm": 0.0, "transmitted": False,
                })
                continue

            rx_j = c.drain_rx(downlink_bits); e_rx_round += rx_j

            local_model = c.local_train(global_model)
            train_compute_j = c.estimated_local_train_energy_j()
            e_compute_round += train_compute_j

            with torch.no_grad():
                vec_l = torch.cat([p.flatten() for p in local_model.parameters()])
                vec_g = torch.cat([p.flatten() for p in global_model.parameters()])
                innov_norm = float(torch.norm(vec_l - vec_g).cpu())
            innov_acc.append(innov_norm)
            innov_compute_j = c.drain_innov_norm_compute(n_params)
            e_aux_round += innov_compute_j

            state = c.q_agent.get_state(c.battery_pct, c.rssi, innov_norm, c.compute_factor)
            e_rl_round += c.drain_q_inference()
            action = c.q_agent.select_action(state, allow_skip=(not is_warmup))
            bits_per_param = BIT_ACTIONS[action]

            if bits_per_param is None:
                c.staleness += 1; n_skipped += 1
                active_t = (c.active_time_for_rx(downlink_bits)
                            + c.active_time_for_compute(train_compute_j + innov_compute_j))
                c.refund_idle_for_active_period(active_t)
                update_records.append({
                    "cid": cid, "state": state, "action": action,
                    "bits_air": 0, "compute_j": train_compute_j, "tx_j": 0.0,
                    "rx_j": rx_j, "aux_j": innov_compute_j,
                    "tx_latency": 0.0, "active_time_s": active_t,
                    "innov_norm": innov_norm, "transmitted": False,
                })
                continue

            quant_compute_j = c.drain_quantize_compute(n_params)
            e_aux_round += quant_compute_j
            q_delta = DeltaQuantizer.quantize_delta(local_model, global_model, bits_per_param)
            if USE_DP:
                q_delta = apply_dp_to_delta(q_delta)

            payload_bits = count_param_bits(local_model, bits_per_param)
            success, bits_air, n_retx = WirelessChannel.transmit(payload_bits, c.rssi)
            n_retx_round += n_retx
            tx_j = EnergyModel.tx_energy_j(bits_air); e_tx_round += tx_j
            tx_latency = bits_air / WIFI_BITRATE_BPS
            c.drain_j(tx_j); round_bits_air += bits_air

            active_t = (c.active_time_for_rx(downlink_bits)
                        + c.active_time_for_compute(train_compute_j + innov_compute_j + quant_compute_j)
                        + c.active_time_for_tx(bits_air))
            c.refund_idle_for_active_period(active_t)

            if success:
                w_innov = 1.0 + min(innov_norm / INNOV_WEIGHT_NORM, 1.0)
                local_deltas.append(q_delta); weights.append(w_innov)
                c.staleness = 0
            else:
                c.staleness += 1

            update_records.append({
                "cid": cid, "state": state, "action": action,
                "bits_air": bits_air,
                "compute_j": train_compute_j, "tx_j": tx_j,
                "rx_j": rx_j, "aux_j": innov_compute_j + quant_compute_j,
                "tx_latency": tx_latency, "active_time_s": active_t,
                "innov_norm": innov_norm,
                "transmitted": success,
            })

        if local_deltas:
            aggregate_deltas(global_model, local_deltas, weights)

        acc, per_class, cm = evaluate(global_model, test_loader)
        per_class_log.append(per_class); cm_final = cm
        acc_delta = acc - prev_acc; prev_acc = acc

        round_rewards = []
        for r in update_records:
            cid = r["cid"]; c = clients[cid]
            reward = compute_reward(r["bits_air"], acc_delta,
                                     r["compute_j"], r["tx_j"], r["tx_latency"])
            round_rewards.append(reward)
            next_state = c.q_agent.get_state(c.battery_pct, c.rssi, r["innov_norm"], c.compute_factor)
            c.q_agent.update(r["state"], r["action"], reward, next_state)
            e_rl_round += c.drain_q_update()
            c.q_agent.decay()
            linucb_r = acc_delta if r["transmitted"] else 0.0
            linucb.update(cid, alive_contexts.get(cid, np.zeros(CONTEXT_DIM)), linucb_r)

        for cid in range(NUM_CLIENTS):
            if cid not in selected_ids and not clients[cid].is_dead:
                clients[cid].staleness += 1

        avg_battery = float(np.mean([c.battery_pct for c in clients]))
        n_dead = sum(1 for c in clients if c.is_dead)
        total_active_s = sum(r.get("active_time_s", 0.0) for r in update_records)
        n_alive = sum(1 for c in clients if not c.is_dead)
        e_idle_gross  = GATEWAY_IDLE_POWER_W * ROUND_WALL_CLOCK_S * n_alive
        e_idle_round  = e_idle_gross - GATEWAY_IDLE_POWER_W * total_active_s
        energy_round = (e_compute_round + e_tx_round + e_rx_round
                         + e_idle_round + e_aux_round + e_rl_round)

        tracker.log_round(rnd, acc, avg_battery, round_bits_air,
                          energy_round, e_compute_round, e_tx_round, e_rx_round,
                          e_idle_round, e_aux_round, e_rl_round,
                          selected_ids, n_skipped, n_dead, n_retx_round,
                          float(np.mean(innov_acc)) if innov_acc else 0.0,
                          float(np.mean(round_rewards)) if round_rewards else 0.0)

        for cid, c in enumerate(clients):
            rec = next((r for r in update_records if r["cid"] == cid), None)
            tracker.log_client_state(rnd, cid, c.battery_pct, c.rssi,
                                     cid in selected_ids,
                                     rec["action"] if rec else None,
                                     rec["bits_air"] if rec else 0,
                                     (rec["compute_j"] + rec["tx_j"]) if rec else 0.0)

        accuracy_log.append(acc * 100); battery_log.append(avg_battery)
        reward_log.append(float(np.mean(round_rewards)) if round_rewards else 0.0)

        if verbose and (rnd <= 10 or rnd % 10 == 0 or rnd == NUM_ROUNDS):
            tag = " [WARMUP]" if is_warmup else ""
            print(f"  R{rnd:3d}{tag:9s} | Acc {acc*100:6.2f}% | "
                  f"E {energy_round:6.2f}J | Bat {avg_battery:5.1f}% | "
                  f"Skip {n_skipped} | Dead {n_dead} | {time.time()-t0:.1f}s")

    return {"accuracy_log": accuracy_log, "battery_log": battery_log,
            "per_class_log": per_class_log, "cm_final": cm_final,
            "reward_log": reward_log,
            "linucb_counts": linucb.selection_counts.copy(),
            "linucb_jain": linucb.jain_fairness(),
            "q_policies": [c.q_agent.policy_table().copy() for c in clients],
            "action_dists": [c.q_agent.get_action_distribution() for c in clients]}


# 16.  FEDAVG BASELINE  (all alive clients, 32-bit, uniform weights)
def run_fedavg_baseline(clients, test_loader, tracker, verbose=True,
                         fedprox_mu=0.0, name="FedAvg"):
    """FedAvg or FedProx baseline (controlled by fedprox_mu).

    fedprox_mu=0.0  → standard FedAvg
    fedprox_mu>0.0  → FedProx with proximal term (Li et al. 2020)
    """
    if verbose:
        print("\n" + "=" * 70)
        print(f"  SIMULATION  -  {name}  (Baseline)   -  {DATASET} / N={NUM_CLIENTS}")
        if fedprox_mu > 0:
            print(f"  FedProx μ = {fedprox_mu}")
        print("=" * 70)

    model_cls = get_model_class()
    global_model = model_cls().to(DEVICE)
    accuracy_log = []; battery_log = []
    per_class_log = []; cm_final = None

    n_params = sum(p.numel() for p in global_model.parameters())

    for rnd in range(1, NUM_ROUNDS + 1):
        t0 = time.time()
        for c in clients: c.step_environment()
        for c in clients:
            if not c.is_dead: c.drain_idle()

        downlink_bits = count_param_bits(global_model, 32)
        local_models = []; round_bits_air = 0
        e_compute_round = 0.0; e_tx_round = 0.0; e_rx_round = 0.0
        n_active = 0; dead_ids = []; n_retx_round = 0; active_times = []

        for c in clients:
            if c.is_dead:
                dead_ids.append(c.id); continue

            rx_j = c.drain_rx(downlink_bits); e_rx_round += rx_j

            local_model = c.local_train(global_model, fedprox_mu=fedprox_mu)
            train_compute_j = c.estimated_local_train_energy_j()
            e_compute_round += train_compute_j

            payload_bits = count_param_bits(local_model, 32)
            success, bits_air, n_retx = WirelessChannel.transmit(payload_bits, c.rssi)
            n_retx_round += n_retx
            tx_j = EnergyModel.tx_energy_j(bits_air); e_tx_round += tx_j
            c.drain_j(tx_j); round_bits_air += bits_air

            active_t = (c.active_time_for_rx(downlink_bits)
                        + c.active_time_for_compute(train_compute_j)
                        + c.active_time_for_tx(bits_air))
            c.refund_idle_for_active_period(active_t)
            active_times.append(active_t)

            if success:
                local_models.append(local_model); n_active += 1

        if local_models:
            fedavg_full_aggregate(global_model, local_models)

        acc, per_class, cm = evaluate(global_model, test_loader)
        per_class_log.append(per_class); cm_final = cm

        avg_battery = float(np.mean([c.battery_pct for c in clients]))
        n_dead = sum(1 for c in clients if c.is_dead)
        n_alive = sum(1 for c in clients if not c.is_dead)
        e_idle_gross = GATEWAY_IDLE_POWER_W * ROUND_WALL_CLOCK_S * n_alive
        e_idle_round = e_idle_gross - GATEWAY_IDLE_POWER_W * sum(active_times)
        energy_round = e_compute_round + e_tx_round + e_rx_round + e_idle_round

        tracker.log_round(rnd, acc, avg_battery, round_bits_air,
                          energy_round, e_compute_round, e_tx_round, e_rx_round,
                          e_idle_round, 0.0, 0.0, list(range(n_active)),
                          0, n_dead, n_retx_round, 0.0, 0.0)

        for cid, c in enumerate(clients):
            tracker.log_client_state(rnd, cid, c.battery_pct, c.rssi,
                                     not c.is_dead, None,
                                     count_param_bits(global_model, 32) if not c.is_dead else 0,
                                     0.0)

        accuracy_log.append(acc * 100); battery_log.append(avg_battery)

        if verbose and (rnd <= 10 or rnd % 10 == 0 or rnd == NUM_ROUNDS):
            print(f"  R{rnd:3d}           | Acc {acc*100:6.2f}% | "
                  f"Active {n_active}/{NUM_CLIENTS} | Dead {n_dead} | "
                  f"E {energy_round:6.2f}J | Bat {avg_battery:5.1f}% | "
                  f"{time.time()-t0:.1f}s")

    return {"accuracy_log": accuracy_log, "battery_log": battery_log,
            "per_class_log": per_class_log, "cm_final": cm_final}


def run_fedprox_baseline(clients, test_loader, tracker, verbose=True):
    return run_fedavg_baseline(clients, test_loader, tracker, verbose=verbose,
                                fedprox_mu=FEDPROX_MU, name="FedProx")


# 17.  OORT BASELINE  (Lai et al. 2021 OSDI  -  utility-based selection)
class OortServer:
    """Implements Oort's utility-based client selection.

    Statistical utility per client:
        U_stat = sqrt(n_samples × avg_loss)   (Lai et al. eq. 1)

    System utility multiplier (penalize stragglers):
        if duration <= T_target: multiplier = 1.0
        else:                    multiplier = (T_target / duration)^alpha

    Total utility = U_stat × multiplier.

    Exploration: ε-greedy over the K slots. With probability ε pick at random
    from all alive clients; otherwise pick top-K by utility.
    """
    def __init__(self, n_clients):
        self.n_clients = n_clients
        self.client_losses = np.zeros(n_clients)
        self.client_n_samples = np.ones(n_clients)
        self.client_durations = np.zeros(n_clients)
        self.target_duration = None
        self.selection_counts = np.zeros(n_clients, dtype=int)

    def update_client(self, cid, loss, n_samples, duration):
        # EMA on loss (factor 0.5 for fast adaptation)
        if self.client_losses[cid] == 0:
            self.client_losses[cid] = loss
        else:
            self.client_losses[cid] = 0.5 * self.client_losses[cid] + 0.5 * loss
        self.client_n_samples[cid] = n_samples
        # EMA on duration
        if self.client_durations[cid] == 0:
            self.client_durations[cid] = duration
        else:
            self.client_durations[cid] = 0.5 * self.client_durations[cid] + 0.5 * duration

    def select(self, alive_ids, k):
        """Select k clients by Oort utility."""
        if not alive_ids:
            return []
        # Determine target duration: median of observed durations (or default to ROUND_WALL_CLOCK_S/2)
        observed = [d for d in self.client_durations if d > 0]
        if observed:
            self.target_duration = float(np.median(observed))
        else:
            self.target_duration = ROUND_WALL_CLOCK_S / 2

        utilities = {}
        for cid in alive_ids:
            loss = self.client_losses[cid]
            n = self.client_n_samples[cid]
            dur = self.client_durations[cid] or self.target_duration
            # Unexplored clients get high exploration boost
            if loss == 0:
                stat_u = 100.0   # large value to ensure exploration
            else:
                stat_u = math.sqrt(max(loss, 0.0) * n)
            if dur <= self.target_duration:
                util = stat_u
            else:
                util = stat_u * (self.target_duration / dur) ** OORT_TIME_ALPHA
            utilities[cid] = util

        # ε-greedy
        if random.random() < OORT_EPSILON or all(self.client_losses[c] == 0 for c in alive_ids):
            selected = random.sample(alive_ids, min(k, len(alive_ids)))
        else:
            ranked = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
            selected = [cid for cid, _ in ranked[:k]]

        for s in selected:
            self.selection_counts[s] += 1
        return selected

    def jain_fairness(self):
        c = self.selection_counts.astype(float)
        if c.sum() == 0: return 0.0
        return float((c.sum() ** 2) / (len(c) * (c ** 2).sum()))


def run_oort_baseline(clients, test_loader, tracker, verbose=True):
    """Oort: utility-based client selection. Uses FedAvg aggregation."""
    if verbose:
        print("\n" + "=" * 70)
        print(f"  SIMULATION  -  Oort  (Utility-Based Selection)   -  {DATASET} / N={NUM_CLIENTS}")
        print("=" * 70)

    model_cls = get_model_class()
    global_model = model_cls().to(DEVICE)
    oort_server = OortServer(NUM_CLIENTS)
    accuracy_log = []; battery_log = []
    per_class_log = []; cm_final = None

    for rnd in range(1, NUM_ROUNDS + 1):
        t0 = time.time()
        for c in clients: c.step_environment()
        for c in clients:
            if not c.is_dead: c.drain_idle()

        alive_ids = [c.id for c in clients if not c.is_dead]
        selected_ids = oort_server.select(alive_ids, K_SELECT)

        downlink_bits = count_param_bits(global_model, 32)
        local_models = []; round_bits_air = 0
        e_compute_round = 0.0; e_tx_round = 0.0; e_rx_round = 0.0
        n_active = 0; n_retx_round = 0; active_times = []

        for cid in selected_ids:
            c = clients[cid]
            if c.is_dead: continue

            rx_j = c.drain_rx(downlink_bits); e_rx_round += rx_j

            t_train = time.time()
            local_model = c.local_train(global_model)
            train_compute_j = c.estimated_local_train_energy_j()
            e_compute_round += train_compute_j
            # Empirical duration in simulated wall-clock seconds for this client
            train_time_simulated = c.active_time_for_compute(train_compute_j)

            payload_bits = count_param_bits(local_model, 32)
            success, bits_air, n_retx = WirelessChannel.transmit(payload_bits, c.rssi)
            n_retx_round += n_retx
            tx_j = EnergyModel.tx_energy_j(bits_air); e_tx_round += tx_j
            c.drain_j(tx_j); round_bits_air += bits_air
            tx_time = c.active_time_for_tx(bits_air)

            active_t = (c.active_time_for_rx(downlink_bits)
                        + train_time_simulated + tx_time)
            c.refund_idle_for_active_period(active_t)
            active_times.append(active_t)

            # Update Oort server with this client's stats
            oort_server.update_client(cid, c.last_local_loss, c.n_samples,
                                       active_t)

            if success:
                local_models.append(local_model); n_active += 1

        if local_models:
            fedavg_full_aggregate(global_model, local_models)

        acc, per_class, cm = evaluate(global_model, test_loader)
        per_class_log.append(per_class); cm_final = cm

        avg_battery = float(np.mean([c.battery_pct for c in clients]))
        n_dead = sum(1 for c in clients if c.is_dead)
        n_alive = sum(1 for c in clients if not c.is_dead)
        e_idle_gross = GATEWAY_IDLE_POWER_W * ROUND_WALL_CLOCK_S * n_alive
        e_idle_round = e_idle_gross - GATEWAY_IDLE_POWER_W * sum(active_times)
        energy_round = e_compute_round + e_tx_round + e_rx_round + e_idle_round

        tracker.log_round(rnd, acc, avg_battery, round_bits_air,
                          energy_round, e_compute_round, e_tx_round, e_rx_round,
                          e_idle_round, 0.0, 0.0, selected_ids,
                          0, n_dead, n_retx_round, 0.0, 0.0)

        for cid, c in enumerate(clients):
            tracker.log_client_state(rnd, cid, c.battery_pct, c.rssi,
                                     cid in selected_ids, None,
                                     count_param_bits(global_model, 32) if cid in selected_ids else 0,
                                     0.0)

        accuracy_log.append(acc * 100); battery_log.append(avg_battery)

        if verbose and (rnd <= 10 or rnd % 10 == 0 or rnd == NUM_ROUNDS):
            print(f"  R{rnd:3d}           | Acc {acc*100:6.2f}% | "
                  f"Sel {len(selected_ids)} | "
                  f"E {energy_round:6.2f}J | Bat {avg_battery:5.1f}% | "
                  f"Dead {n_dead} | {time.time()-t0:.1f}s")

    return {"accuracy_log": accuracy_log, "battery_log": battery_log,
            "per_class_log": per_class_log, "cm_final": cm_final,
            "oort_counts": oort_server.selection_counts.copy(),
            "oort_jain": oort_server.jain_fairness()}


# 18.  MULTI-SEED + MULTI-BASELINE ORCHESTRATOR
def make_clients(train_loaders, n_samples_per_client,
                 init_batteries, init_rssi, init_compute):
    """Build a fresh list of IoTClients with the given initial state."""
    return [IoTClient(i, train_loaders[i], n_samples_per_client[i],
                       battery_pct=init_batteries[i],
                       rssi=init_rssi[i],
                       compute_factor=init_compute[i])
            for i in range(NUM_CLIENTS)]


def run_single_seed(seed, verbose=False):
    set_global_seed(seed)
    train_loaders, test_loader, n_samples = get_dataset_loaders(seed)

    init_batteries = [random.uniform(60, 100) for _ in range(NUM_CLIENTS)]
    init_rssi      = [random.uniform(-65, -45) for _ in range(NUM_CLIENTS)]
    init_compute   = [random.uniform(0.5, 1.5) for _ in range(NUM_CLIENTS)]

    # Hybrid-RL
    clients_hybrid = make_clients(train_loaders, n_samples,
                                    init_batteries, init_rssi, init_compute)
    tracker_hybrid = MetricsTracker("Hybrid-RL")

    # Snapshot RNG so each baseline sees same channel trajectory
    snap = snapshot_rng_state()
    hybrid = run_hybrid_rl(clients_hybrid, test_loader, tracker_hybrid, verbose=verbose)

    baseline_results = {}
    baseline_trackers = {"Hybrid-RL": tracker_hybrid}

    for baseline_name in BASELINES_TO_RUN:
        restore_rng_state(snap)
        clients_b = make_clients(train_loaders, n_samples,
                                  init_batteries, init_rssi, init_compute)
        tracker_b = MetricsTracker(baseline_name.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg"))
        if baseline_name == "fedavg":
            result = run_fedavg_baseline(clients_b, test_loader, tracker_b, verbose=verbose)
        elif baseline_name == "fedprox":
            result = run_fedprox_baseline(clients_b, test_loader, tracker_b, verbose=verbose)
        elif baseline_name == "oort":
            result = run_oort_baseline(clients_b, test_loader, tracker_b, verbose=verbose)
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")
        baseline_results[baseline_name] = result
        baseline_trackers[tracker_b.system_name] = tracker_b

    return baseline_trackers, hybrid, baseline_results


def run_multi_seed(seeds=SEEDS):
    print(f"\n[MULTI-SEED] Running {len(seeds)} seeds: {seeds}")
    print(f"[INFO] Device: {DEVICE} | Dataset: {DATASET} | N_CLIENTS: {NUM_CLIENTS} "
          f"| K_SELECT: {K_SELECT} | Q-agent: {Q_AGENT_TYPE}")
    print(f"[INFO] Baselines: Hybrid-RL + {BASELINES_TO_RUN}")

    all_results = {
        "hybrid": [],
        "trackers": [],            # list of dicts (one per seed) {system_name: tracker}
        "baseline_results": [],    # list of dicts {baseline_name: result}
        "seeds": list(seeds),
    }

    for i, s in enumerate(seeds):
        print(f"\n{'#'*72}")
        print(f"#  SEED {s}   ({i+1}/{len(seeds)})")
        print(f"{'#'*72}", flush=True)
        t0 = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trackers, hybrid, baseline_results = run_single_seed(s, verbose=True)

        all_results["hybrid"].append(hybrid)
        all_results["trackers"].append(trackers)
        all_results["baseline_results"].append(baseline_results)

        # Save per-seed CSVs
        seed_dir = OUT_DIR / f"seed_{s}"
        seed_dir.mkdir(exist_ok=True)
        for tracker in trackers.values():
            tracker.save_all(seed_dir)
        for baseline_name in BASELINES_TO_RUN:
            save_config(s, baseline_name, seed_dir / f"config_{baseline_name}.json")
        save_config(s, "hybrid", seed_dir / "config_hybrid.json")
        print(f"\n  [SEED {s}] done in {(time.time()-t0)/60:.1f} min", flush=True)

    return all_results


# 19.  STATISTICAL ANALYSIS  (4-way comparison)
def statistical_comparison(results):
    """Compute paired t-test and Mann-Whitney U for Hybrid-RL vs each baseline."""
    h_acc = [r["accuracy_log"][-1] for r in results["hybrid"]]
    h_energy = [t["Hybrid-RL"]._energy_cum for t in results["trackers"]]
    h_bits = [t["Hybrid-RL"]._bits_cum for t in results["trackers"]]

    rows = []
    for baseline_name in BASELINES_TO_RUN:
        b_acc = [br[baseline_name]["accuracy_log"][-1] for br in results["baseline_results"]]
        sys_name = baseline_name.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        b_energy = [t[sys_name]._energy_cum for t in results["trackers"]]
        b_bits = [t[sys_name]._bits_cum for t in results["trackers"]]

        t_acc, p_acc = stats.ttest_rel(h_acc, b_acc)
        try:
            u_acc, pu_acc = stats.mannwhitneyu(h_acc, b_acc, alternative="two-sided")
            mw_str = f"{u_acc:.1f}"; pu_str = f"{pu_acc:.4f}"
        except Exception:
            mw_str = "—"; pu_str = "—"
        t_e, p_e = stats.ttest_rel(h_energy, b_energy)
        t_b, p_b = stats.ttest_rel(h_bits, b_bits)

        rows.append({
            "Metric": f"Accuracy: Hybrid-RL vs {sys_name}",
            "Hybrid-RL μ ± σ": f"{np.mean(h_acc):.2f} ± {np.std(h_acc):.2f}",
            f"{sys_name} μ ± σ": f"{np.mean(b_acc):.2f} ± {np.std(b_acc):.2f}",
            "Paired t": f"{t_acc:.3f}", "p (t)": f"{p_acc:.4f}",
            "MW-U": mw_str, "p (MW)": pu_str,
        })
        rows.append({
            "Metric": f"Energy: Hybrid-RL vs {sys_name}",
            "Hybrid-RL μ ± σ": f"{np.mean(h_energy):.1f} ± {np.std(h_energy):.1f}",
            f"{sys_name} μ ± σ": f"{np.mean(b_energy):.1f} ± {np.std(b_energy):.1f}",
            "Paired t": f"{t_e:.3f}", "p (t)": f"{p_e:.4f}",
            "MW-U": "—", "p (MW)": "—",
        })
        rows.append({
            "Metric": f"Comm: Hybrid-RL vs {sys_name}",
            "Hybrid-RL μ ± σ": f"{np.mean(h_bits)/1e9:.3f} ± {np.std(h_bits)/1e9:.3f}",
            f"{sys_name} μ ± σ": f"{np.mean(b_bits)/1e9:.3f} ± {np.std(b_bits)/1e9:.3f}",
            "Paired t": f"{t_b:.3f}", "p (t)": f"{p_b:.4f}",
            "MW-U": "—", "p (MW)": "—",
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "statistical_tests.csv", index=False)
    return df


# 20.  VISUALIZATIONS  (4-way comparison)
PLOT_THEME = {
    "C_HYBRID": "#2ECC71", "C_FEDAVG": "#E74C3C",
    "C_FEDPROX": "#3498DB", "C_OORT": "#F1C40F",
    "BG": "#0F1117", "AXBG": "#16181F",
    "GRID": "#2A2D35", "TEXT": "#EAEAEA",
}

BASELINE_COLORS = {
    "Hybrid-RL": PLOT_THEME["C_HYBRID"],
    "FedAvg":    PLOT_THEME["C_FEDAVG"],
    "FedProx":   PLOT_THEME["C_FEDPROX"],
    "Oort":      PLOT_THEME["C_OORT"],
}


def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor": PLOT_THEME["BG"],
        "axes.facecolor":   PLOT_THEME["AXBG"],
        "axes.edgecolor":   PLOT_THEME["GRID"],
        "axes.labelcolor":  PLOT_THEME["TEXT"],
        "xtick.color":      PLOT_THEME["TEXT"],
        "ytick.color":      PLOT_THEME["TEXT"],
        "text.color":       PLOT_THEME["TEXT"],
        "grid.color":       PLOT_THEME["GRID"],
        "grid.linestyle":   "--", "grid.alpha": 0.5,
        "legend.facecolor": "#1E2029", "legend.edgecolor": PLOT_THEME["GRID"],
        "font.family":      "monospace",
        "axes.titlesize":   12, "axes.labelsize": 11,
    })


def _save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=150, bbox_inches="tight",
                    facecolor=PLOT_THEME["BG"])
    plt.close(fig)


def _all_systems(results):
    """Return list of (system_name, accuracy_array_per_seed)."""
    sys_data = {"Hybrid-RL": np.array([r["accuracy_log"] for r in results["hybrid"]])}
    for bname in BASELINES_TO_RUN:
        sys_name = bname.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        sys_data[sys_name] = np.array([br[bname]["accuracy_log"] for br in results["baseline_results"]])
    return sys_data


def fig01_accuracy_curves(results):
    _apply_theme()
    rounds = np.arange(1, NUM_ROUNDS + 1)
    sys_data = _all_systems(results)

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=PLOT_THEME["BG"])
    fig.suptitle(f"① Global Accuracy vs FL Round  -  4-way comparison ({DATASET}, N={NUM_CLIENTS}, 5 seeds, 95% CI)",
                 fontsize=13, fontweight="bold", color=PLOT_THEME["TEXT"])
    for sys_name, arr in sys_data.items():
        color = BASELINE_COLORS.get(sys_name, "#888")
        m = arr.mean(0); s = arr.std(0)
        ci = 1.96 * s / max(math.sqrt(arr.shape[0]), 1)
        ax.plot(rounds, m, color=color, lw=2.2, label=sys_name)
        ax.fill_between(rounds, m - ci, m + ci, color=color, alpha=0.15)
    ax.axvline(WARMUP_ROUNDS, color="#F39C12", ls=":", alpha=0.7)
    ax.set(xlabel="Round", ylabel="Global Test Accuracy (%)",
            xlim=(1, NUM_ROUNDS))
    ax.legend(loc="lower right"); ax.grid(True)
    _save_fig(fig, "fig01_accuracy_4way")


def fig02_energy_cumulative(results):
    _apply_theme()
    rounds = np.arange(1, NUM_ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=PLOT_THEME["BG"])
    fig.suptitle(f"② Cumulative Energy Consumption  -  4-way comparison ({DATASET}, N={NUM_CLIENTS})",
                 fontsize=13, fontweight="bold", color=PLOT_THEME["TEXT"])
    # Hybrid
    h_energy = np.array([t["Hybrid-RL"].energy_log for t in results["trackers"]])
    m = h_energy.mean(0); s = h_energy.std(0)
    ax.plot(rounds, m, color=PLOT_THEME["C_HYBRID"], lw=2.2, label="Hybrid-RL")
    ax.fill_between(rounds, m-s, m+s, color=PLOT_THEME["C_HYBRID"], alpha=0.15)
    # Baselines
    for bname in BASELINES_TO_RUN:
        sys_name = bname.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        b_energy = np.array([t[sys_name].energy_log for t in results["trackers"]])
        m = b_energy.mean(0); s = b_energy.std(0)
        color = BASELINE_COLORS.get(sys_name, "#888")
        ax.plot(rounds, m, color=color, lw=2.2, label=sys_name)
        ax.fill_between(rounds, m-s, m+s, color=color, alpha=0.15)
    ax.set(xlabel="Round", ylabel="Cumulative Energy (J)",
            xlim=(1, NUM_ROUNDS))
    ax.legend(loc="upper left"); ax.grid(True)
    _save_fig(fig, "fig02_energy_4way")


def fig03_comm_cumulative(results):
    _apply_theme()
    rounds = np.arange(1, NUM_ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=PLOT_THEME["BG"])
    fig.suptitle(f"③ Cumulative Communication  -  4-way comparison ({DATASET}, N={NUM_CLIENTS})",
                 fontsize=13, fontweight="bold", color=PLOT_THEME["TEXT"])
    h_bits = np.array([t["Hybrid-RL"].bits_log for t in results["trackers"]]) / 1e9
    m = h_bits.mean(0); s = h_bits.std(0)
    ax.plot(rounds, m, color=PLOT_THEME["C_HYBRID"], lw=2.2, label="Hybrid-RL")
    ax.fill_between(rounds, m-s, m+s, color=PLOT_THEME["C_HYBRID"], alpha=0.15)
    for bname in BASELINES_TO_RUN:
        sys_name = bname.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        b_bits = np.array([t[sys_name].bits_log for t in results["trackers"]]) / 1e9
        m = b_bits.mean(0); s = b_bits.std(0)
        color = BASELINE_COLORS.get(sys_name, "#888")
        ax.plot(rounds, m, color=color, lw=2.2, label=sys_name)
        ax.fill_between(rounds, m-s, m+s, color=color, alpha=0.15)
    ax.set(xlabel="Round", ylabel="Cumulative Communication (Gbit)",
            xlim=(1, NUM_ROUNDS))
    ax.legend(loc="upper left"); ax.grid(True)
    _save_fig(fig, "fig03_communication_4way")


def fig04_final_acc_box(results):
    _apply_theme()
    sys_data = _all_systems(results)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PLOT_THEME["BG"])
    fig.suptitle(f"④ Final-Round Accuracy Distribution ({DATASET}, N={NUM_CLIENTS}, 5 seeds)",
                 fontsize=13, fontweight="bold", color=PLOT_THEME["TEXT"])
    labels = list(sys_data.keys())
    data = [sys_data[k][:, -1] for k in labels]
    colors = [BASELINE_COLORS.get(l, "#888") for l in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
        patch.set_edgecolor(PLOT_THEME["TEXT"])
    for median in bp['medians']:
        median.set_color(PLOT_THEME["TEXT"]); median.set_linewidth(2)
    ax.set(ylabel="Final Accuracy (%)")
    ax.grid(True, axis="y")
    _save_fig(fig, "fig04_final_acc_box")


def fig05_pareto_4way(results):
    _apply_theme()
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=PLOT_THEME["BG"])
    fig.suptitle(f"⑤ Energy-Accuracy Pareto Trajectory  -  4-way ({DATASET}, N={NUM_CLIENTS})",
                 fontsize=13, fontweight="bold", color=PLOT_THEME["TEXT"])
    # Hybrid
    h_energy = np.array([t["Hybrid-RL"].energy_log for t in results["trackers"]])
    h_acc = np.array([r["accuracy_log"] for r in results["hybrid"]])
    em = h_energy.mean(0); am = h_acc.mean(0)
    ax.plot(em, am, color=PLOT_THEME["C_HYBRID"], lw=2, label="Hybrid-RL",
            marker="o", markersize=3, alpha=0.85)
    ax.scatter(em[-1], am[-1], color=PLOT_THEME["C_HYBRID"], s=120, zorder=4,
               edgecolor="black")
    for bname in BASELINES_TO_RUN:
        sys_name = bname.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        b_energy = np.array([t[sys_name].energy_log for t in results["trackers"]])
        b_acc = np.array([br[bname]["accuracy_log"] for br in results["baseline_results"]])
        em = b_energy.mean(0); am = b_acc.mean(0)
        color = BASELINE_COLORS.get(sys_name, "#888")
        ax.plot(em, am, color=color, lw=2, label=sys_name,
                marker="s", markersize=3, alpha=0.85)
        ax.scatter(em[-1], am[-1], color=color, s=120, zorder=4,
                   edgecolor="black")
    ax.set(xlabel="Cumulative Energy (J)", ylabel="Global Accuracy (%)")
    ax.legend(loc="lower right"); ax.grid(True)
    _save_fig(fig, "fig05_pareto_4way")


def fig06_battery_curves(results):
    _apply_theme()
    rounds = np.arange(1, NUM_ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=PLOT_THEME["BG"])
    fig.suptitle(f"⑥ Average Client Battery  -  4-way ({DATASET}, N={NUM_CLIENTS})",
                 fontsize=13, fontweight="bold", color=PLOT_THEME["TEXT"])
    h_bat = np.array([r["battery_log"] for r in results["hybrid"]])
    m = h_bat.mean(0); s = h_bat.std(0)
    ax.plot(rounds, m, color=PLOT_THEME["C_HYBRID"], lw=2.2, label="Hybrid-RL")
    ax.fill_between(rounds, m-s, m+s, color=PLOT_THEME["C_HYBRID"], alpha=0.15)
    for bname in BASELINES_TO_RUN:
        sys_name = bname.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        b_bat = np.array([br[bname]["battery_log"] for br in results["baseline_results"]])
        m = b_bat.mean(0); s = b_bat.std(0)
        color = BASELINE_COLORS.get(sys_name, "#888")
        ax.plot(rounds, m, color=color, lw=2.2, label=sys_name)
        ax.fill_between(rounds, m-s, m+s, color=color, alpha=0.15)
    ax.axhline(20, color="#F39C12", ls=":", lw=1.2, label="Critical (20%)")
    ax.set(xlabel="Round", ylabel="Average Battery (%)", xlim=(1, NUM_ROUNDS),
           ylim=(0, 105))
    ax.legend(loc="lower left"); ax.grid(True)
    _save_fig(fig, "fig06_battery_4way")


def fig07_savings_bars(results):
    _apply_theme()
    h_energy = np.mean([t["Hybrid-RL"]._energy_cum for t in results["trackers"]])
    h_bits = np.mean([t["Hybrid-RL"]._bits_cum for t in results["trackers"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PLOT_THEME["BG"])
    fig.suptitle(f"⑦ Hybrid-RL Savings vs Each Baseline ({DATASET}, N={NUM_CLIENTS})",
                 fontsize=13, fontweight="bold", color=PLOT_THEME["TEXT"])

    baselines_for_plot = []
    energy_savings = []
    comm_savings = []
    for bname in BASELINES_TO_RUN:
        sys_name = bname.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        b_energy = np.mean([t[sys_name]._energy_cum for t in results["trackers"]])
        b_bits = np.mean([t[sys_name]._bits_cum for t in results["trackers"]])
        baselines_for_plot.append(sys_name)
        energy_savings.append((1 - h_energy / b_energy) * 100)
        comm_savings.append((1 - h_bits / b_bits) * 100)

    colors = [BASELINE_COLORS.get(b, "#888") for b in baselines_for_plot]
    x = np.arange(len(baselines_for_plot))

    axes[0].bar(x, energy_savings, color=colors, edgecolor="black", lw=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(baselines_for_plot)
    axes[0].set(ylabel="Energy Savings (%)", title="Energy savings vs each baseline")
    for i, v in enumerate(energy_savings):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", color=PLOT_THEME["TEXT"])
    axes[0].grid(True, axis="y")

    axes[1].bar(x, comm_savings, color=colors, edgecolor="black", lw=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(baselines_for_plot)
    axes[1].set(ylabel="Communication Savings (%)", title="Communication savings vs each baseline")
    for i, v in enumerate(comm_savings):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", color=PLOT_THEME["TEXT"])
    axes[1].grid(True, axis="y")
    _save_fig(fig, "fig07_savings_bars")


def fig08_per_seed_acc(results):
    _apply_theme()
    sys_data = _all_systems(results)
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=PLOT_THEME["BG"])
    fig.suptitle(f"⑧ Per-Seed Final Accuracy ({DATASET}, N={NUM_CLIENTS})",
                 fontsize=13, fontweight="bold", color=PLOT_THEME["TEXT"])
    width = 0.18
    x = np.arange(len(results["seeds"]))
    for i, (sys_name, arr) in enumerate(sys_data.items()):
        color = BASELINE_COLORS.get(sys_name, "#888")
        finals = arr[:, -1]
        ax.bar(x + i * width, finals, width=width, color=color,
               edgecolor="black", lw=0.4, label=sys_name)
    ax.set_xticks(x + width * (len(sys_data) - 1) / 2)
    ax.set_xticklabels([str(s) for s in results["seeds"]])
    ax.set(xlabel="Seed", ylabel="Final Accuracy (%)", ylim=(0, 105))
    ax.legend(); ax.grid(True, axis="y")
    _save_fig(fig, "fig08_per_seed_acc")


def generate_all_figures(results):
    print("\n[VIZ] Generating publication figures...")
    fig01_accuracy_curves(results);    print("  ✓ fig01_accuracy_4way")
    fig02_energy_cumulative(results);  print("  ✓ fig02_energy_4way")
    fig03_comm_cumulative(results);    print("  ✓ fig03_communication_4way")
    fig04_final_acc_box(results);      print("  ✓ fig04_final_acc_box")
    fig05_pareto_4way(results);        print("  ✓ fig05_pareto_4way")
    fig06_battery_curves(results);     print("  ✓ fig06_battery_4way")
    fig07_savings_bars(results);       print("  ✓ fig07_savings_bars")
    fig08_per_seed_acc(results);       print("  ✓ fig08_per_seed_acc")
    print(f"[VIZ] All figures saved → {FIG_DIR}/")


# 21.  SUMMARY TABLES
def print_and_save_summary(results):
    sys_data = _all_systems(results)
    rows = []

    # Final accuracy
    row = {"Metric": "Final Accuracy (%)"}
    for sys_name, arr in sys_data.items():
        finals = arr[:, -1]
        row[sys_name] = f"{np.mean(finals):.2f} ± {np.std(finals):.2f}"
    rows.append(row)

    # Total Comm
    row = {"Metric": "Total Comm (Gbit)"}
    row["Hybrid-RL"] = f"{np.mean([t['Hybrid-RL']._bits_cum for t in results['trackers']])/1e9:.3f}"
    for bname in BASELINES_TO_RUN:
        sys_name = bname.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        row[sys_name] = f"{np.mean([t[sys_name]._bits_cum for t in results['trackers']])/1e9:.3f}"
    rows.append(row)

    # Total Energy
    row = {"Metric": "Total Energy (J)"}
    row["Hybrid-RL"] = f"{np.mean([t['Hybrid-RL']._energy_cum for t in results['trackers']]):.1f}"
    for bname in BASELINES_TO_RUN:
        sys_name = bname.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        row[sys_name] = f"{np.mean([t[sys_name]._energy_cum for t in results['trackers']]):.1f}"
    rows.append(row)

    # Battery remaining
    row = {"Metric": "Battery Remaining (%)"}
    row["Hybrid-RL"] = f"{np.mean([r['battery_log'][-1] for r in results['hybrid']]):.1f}"
    for bname in BASELINES_TO_RUN:
        sys_name = bname.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        bats = [br[bname]["battery_log"][-1] for br in results["baseline_results"]]
        row[sys_name] = f"{np.mean(bats):.1f}"
    rows.append(row)

    # Comm / Energy savings vs each baseline
    h_e = np.mean([t["Hybrid-RL"]._energy_cum for t in results["trackers"]])
    h_b = np.mean([t["Hybrid-RL"]._bits_cum for t in results["trackers"]])
    for bname in BASELINES_TO_RUN:
        sys_name = bname.capitalize().replace("Fedprox", "FedProx").replace("Fedavg", "FedAvg")
        b_e = np.mean([t[sys_name]._energy_cum for t in results["trackers"]])
        b_b = np.mean([t[sys_name]._bits_cum for t in results["trackers"]])
        comm_save = (1 - h_b / b_b) * 100 if b_b > 0 else 0
        e_save = (1 - h_e / b_e) * 100 if b_e > 0 else 0
        row = {"Metric": f"Comm savings vs {sys_name} (%)"}
        for s in sys_data.keys(): row[s] = "—"
        row["Hybrid-RL"] = f"{comm_save:.1f}"
        rows.append(row)
        row = {"Metric": f"Energy savings vs {sys_name} (%)"}
        for s in sys_data.keys(): row[s] = "—"
        row["Hybrid-RL"] = f"{e_save:.1f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "final_summary.csv", index=False)

    print("\n" + "=" * 90)
    print(f"  FINAL RESULTS  -  {DATASET} / N={NUM_CLIENTS} / Q-agent={Q_AGENT_TYPE} "
          f"(mean ± std over seeds)")
    print("=" * 90)
    print(df.to_string(index=False))

    stat_df = statistical_comparison(results)
    print("\n[STATS] Significance tests (Hybrid-RL vs each baseline):")
    print(stat_df.to_string(index=False))


# 22.  MAIN
def main():
    print("=" * 80)
    print("  ABFB - Federated Learning baseline runner")
    print(f"  Dataset        : {DATASET.upper()}")
    print(f"  Clients        : {NUM_CLIENTS}  (K_SELECT = {K_SELECT})")
    print(f"  Q-agent variant: {Q_AGENT_TYPE}")
    print(f"  Baselines      : Hybrid-RL + {BASELINES_TO_RUN}")
    print(f"  Device         : {DEVICE}")
    print(f"  Seeds          : {SEEDS}")
    print(f"  Output dir     : {OUT_DIR.resolve()}")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")

    t_total = time.time()
    results = run_multi_seed(SEEDS)
    elapsed = (time.time() - t_total) / 60
    print(f"\n[TIME] Total wall-clock: {elapsed:.1f} min")

    print_and_save_summary(results)
    generate_all_figures(results)

    print("\n[DONE] All outputs:")
    for p in sorted(OUT_DIR.iterdir()):
        if p.is_file():
            print(f"   {p.name}  ({p.stat().st_size // 1024} KB)")
        else:
            print(f"   {p.name}/")


if __name__ == "__main__":
    main()
