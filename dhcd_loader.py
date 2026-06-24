"""
================================================================================
  DHCD (Devanagari Handwritten Character Dataset) loader for ABFB FL.

  Source folder structure (already on disk; not downloaded by this module):
      <root>\\Train\\character_X_name\\*.png
      <root>\\Train\\digit_X\\*.png
      <root>\\Test\\<class_folder>\\*.png

  Stats: 46 classes (36 consonants + 10 digits), 32×32 grayscale PNG,
         1,700 train + 300 test images per class. Total 78,200 train / 13,800 test.

  Path discovery (in priority order):
      1. DHCD_ROOT environment variable, if set
      2. <cwd>\\DHCD                  (when running from the project folder)
      3. <script_parent>\\DHCD        (sibling to fl_simulation_phase3.py)
      4. <script_parent_parent>\\DHCD (sibling to phase3_active_belief/)
      5. D:\\Pranta\\THESIS\\Code\\DHCD (legacy default)

  Performance:
      All images are decoded ONCE at first call and held as a single uint8
      tensor in RAM (≈ 80 MB train + 14 MB test). Subsequent iterations are
      zero-disk-read. This makes DHCD ~10–20× faster than ImageFolder per epoch.

  Partition: Dirichlet(α) non-IID across N clients (matches baseline HAR loader).
================================================================================
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


# Constants
DHCD_N_CLASSES   = 46
DHCD_IMG_SIZE    = 32
DHCD_MEAN        = 0.06     # empirical mean of normalized 32×32 grayscale DHCD
DHCD_STD         = 0.23

# Candidate roots to try, in priority order.
def _candidate_roots() -> List[Path]:
    here = Path(__file__).resolve().parent
    roots: List[Path] = []
    env = os.environ.get("DHCD_ROOT")
    if env:
        roots.append(Path(env))
    roots.extend([
        Path.cwd() / "DHCD",
        here / "DHCD",
        here.parent / "DHCD",
        here.parent.parent / "DHCD",
        Path("D:/Pranta/THESIS/Code/DHCD"),
    ])
    return roots


def _discover_root() -> Path:
    """Find DHCD on disk; raise if none of the candidates works."""
    for r in _candidate_roots():
        if (r / "Train").exists() and (r / "Test").exists():
            return r
    raise FileNotFoundError(
        "Could not locate DHCD. Set DHCD_ROOT env var to the directory "
        "containing Train/ and Test/ subfolders, or place DHCD/ in the "
        "current working directory."
    )


DHCD_DEFAULT_DIR = None     # resolved lazily in _get_train_test()


# In-memory dataset (decode once, slice forever)
class _InMemoryDHCD(Dataset):
    """Decoded uint8 tensor + labels, fully resident in RAM."""

    def __init__(self, images_u8: torch.Tensor, labels: np.ndarray):
        # images_u8: (N, 32, 32) uint8 tensor
        self.images_u8 = images_u8
        self.labels = labels
        # Pre-compute float normalization constants (avoid per-getitem allocation)
        self._mean = DHCD_MEAN
        self._inv_std = 1.0 / DHCD_STD

    def __len__(self):
        return self.images_u8.shape[0]

    def __getitem__(self, idx):
        # Convert uint8 → float32 in [0,1], normalize, add channel dim
        img = self.images_u8[idx].float().div_(255.0)
        img = (img - self._mean) * self._inv_std
        return img.unsqueeze(0), int(self.labels[idx])


def _scan_and_decode(split_root: Path) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    """
    Walk split_root/<class_folder>/*.png, decode each into a uint8 (32,32)
    tensor, return stacked tensor + label array + class-name list.
    """
    class_dirs = sorted(p for p in split_root.iterdir() if p.is_dir())
    classes = [d.name for d in class_dirs]

    all_imgs: List[np.ndarray] = []
    all_labels: List[int] = []

    for cls_idx, cls_dir in enumerate(class_dirs):
        for png_path in sorted(cls_dir.iterdir()):
            if png_path.suffix.lower() != ".png":
                continue
            with Image.open(png_path) as im:
                im = im.convert("L")
                if im.size != (DHCD_IMG_SIZE, DHCD_IMG_SIZE):
                    im = im.resize((DHCD_IMG_SIZE, DHCD_IMG_SIZE))
                arr = np.asarray(im, dtype=np.uint8)
            all_imgs.append(arr)
            all_labels.append(cls_idx)

    images_u8 = torch.from_numpy(np.stack(all_imgs, axis=0))   # (N, 32, 32) uint8
    labels = np.asarray(all_labels, dtype=np.int64)
    return images_u8, labels, classes


# Cached in-memory datasets (decoded once per process)
_TRAIN_DS_CACHE: _InMemoryDHCD | None = None
_TEST_DS_CACHE:  _InMemoryDHCD | None = None
_TRAIN_LABELS_CACHE: np.ndarray | None = None
_ROOT_USED: Path | None = None


def _get_train_test(root: Path | None) -> Tuple[_InMemoryDHCD, _InMemoryDHCD, np.ndarray]:
    """Return cached datasets; first call decodes all PNGs into RAM."""
    global _TRAIN_DS_CACHE, _TEST_DS_CACHE, _TRAIN_LABELS_CACHE, _ROOT_USED

    if root is None:
        root = _discover_root()

    if _TRAIN_DS_CACHE is None or _ROOT_USED != root:
        print(f"[DHCD] First-use decode from {root}/  (one-time, ~3-5 s) ...")
        train_imgs, train_labels, classes = _scan_and_decode(root / "Train")
        assert len(classes) == DHCD_N_CLASSES, (
            f"Train: expected {DHCD_N_CLASSES} classes, got {len(classes)}")
        test_imgs, test_labels, test_classes = _scan_and_decode(root / "Test")
        assert len(test_classes) == DHCD_N_CLASSES, (
            f"Test: expected {DHCD_N_CLASSES} classes, got {len(test_classes)}")
        assert classes == test_classes, (
            "Train/Test class folder lists differ  -  label mapping would be wrong")

        _TRAIN_DS_CACHE = _InMemoryDHCD(train_imgs, train_labels)
        _TEST_DS_CACHE  = _InMemoryDHCD(test_imgs, test_labels)
        _TRAIN_LABELS_CACHE = train_labels
        _ROOT_USED = root
        n_train = train_imgs.shape[0]
        n_test  = test_imgs.shape[0]
        mb = (train_imgs.numel() + test_imgs.numel()) / (1024 * 1024)
        print(f"[DHCD] Loaded {n_train} train + {n_test} test samples "
              f"({mb:.1f} MB resident).")

    return _TRAIN_DS_CACHE, _TEST_DS_CACHE, _TRAIN_LABELS_CACHE


def build_dhcd_loaders(
    num_clients: int,
    batch_size: int,
    seed: int,
    dirichlet_alpha: float = 0.3,
    root: Path | None = None,
) -> Tuple[List[DataLoader], DataLoader, List[int]]:
    """
    Dirichlet(α) non-IID partition of DHCD training set across `num_clients`.

    Returns
    -------
    client_loaders : list of DataLoader, one per client
    test_loader    : DataLoader over the full test split (256 batch)
    n_samples_per_client : per-client training-sample count (for FL weighting)

    Determinism: fixed seed → identical partition across runs of this loader.
    Performance: first call decodes all PNGs to RAM (~3–5 s); subsequent
    calls are zero-disk-read.
    """
    train_ds, test_ds, train_labels = _get_train_test(root)
    np_rng = np.random.RandomState(seed)

    # Dirichlet per-class partition (Hsu et al. 2019)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for cls in range(DHCD_N_CLASSES):
        cls_idx = np.where(train_labels == cls)[0]
        np_rng.shuffle(cls_idx)
        props = np_rng.dirichlet(np.ones(num_clients) * dirichlet_alpha)
        counts = (props * len(cls_idx)).astype(int)
        counts[-1] = len(cls_idx) - counts[:-1].sum()    # absorb rounding
        start = 0
        for c, n in enumerate(counts):
            if n > 0:
                client_indices[c].extend(cls_idx[start:start + n].tolist())
                start += n

    # Build loaders
    client_loaders: List[DataLoader] = []
    n_samples_per_client: List[int] = []
    for c in range(num_clients):
        idx = client_indices[c]
        if len(idx) == 0:
            idx = [0]    # avoid empty loader
        loader = DataLoader(
            Subset(train_ds, idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,            # in-memory, no benefit from workers
            pin_memory=False,
        )
        client_loaders.append(loader)
        n_samples_per_client.append(len(idx))

    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    return client_loaders, test_loader, n_samples_per_client


# Quick smoke test
if __name__ == "__main__":
    print("DHCD loader smoke test")
    root = _discover_root()
    print(f"  Root resolved to: {root}")
    loaders, test_loader, ns = build_dhcd_loaders(
        num_clients=10, batch_size=32, seed=42, dirichlet_alpha=0.3)
    print(f"  Built {len(loaders)} client loaders")
    print(f"  Per-client sample counts: {ns}")
    print(f"  Total train samples: {sum(ns)}")
    print(f"  Test loader: {len(test_loader.dataset)} samples")
    x, y = next(iter(loaders[0]))
    print(f"  Client 0 first batch: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}")
    print("Smoke OK.")
