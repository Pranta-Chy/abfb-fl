"""In-memory CIFAR-10 loader with Dirichlet(alpha=0.3) non-IID partitions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10


# Constants
CIFAR10_N_CLASSES = 10
CIFAR10_IMG_SIZE  = 32
# Empirical CIFAR-10 channel-wise mean/std (standard torchvision values)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def _candidate_roots() -> List[Path]:
    """Candidate locations for the CIFAR-10 cache directory, in priority order."""
    here = Path(__file__).resolve().parent
    roots: List[Path] = []
    env = os.environ.get("CIFAR10_ROOT")
    if env:
        roots.append(Path(env))
    roots.extend([
        Path.cwd() / "cifar10",
        here / "cifar10",
        here.parent / "cifar10",
        here.parent.parent / "cifar10",
        Path("D:/Pranta/THESIS/Code/cifar10"),
    ])
    return roots


def _discover_root(allow_download: bool = True) -> Path:
    """
    Find an existing CIFAR-10 cache directory; if none, create the first
    candidate (when allow_download=True) so torchvision can populate it.
    """
    for r in _candidate_roots():
        cifar_dir = r / "cifar-10-batches-py"
        if cifar_dir.exists():
            return r
    if not allow_download:
        raise FileNotFoundError(
            "Could not locate cached CIFAR-10. Set CIFAR10_ROOT or place "
            "cifar-10-batches-py/ in one of the standard locations."
        )
    # Create the first candidate so torchvision can download into it
    target = _candidate_roots()[0]
    target.mkdir(parents=True, exist_ok=True)
    return target


# In-memory dataset
class _InMemoryCIFAR10(Dataset):
    """Decoded uint8 tensor (N, 3, 32, 32) + label array, fully resident in RAM."""

    def __init__(self, images_u8: torch.Tensor, labels: np.ndarray):
        # images_u8: (N, 3, 32, 32) uint8
        self.images_u8 = images_u8
        self.labels = labels
        # Pre-compute per-channel normalization (1, 3, 1, 1) for broadcasting
        self._mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        self._inv_std = (1.0 / torch.tensor(CIFAR10_STD)).view(3, 1, 1)

    def __len__(self) -> int:
        return self.images_u8.shape[0]

    def __getitem__(self, idx):
        img = self.images_u8[idx].float().div_(255.0)
        img = (img - self._mean) * self._inv_std
        return img, int(self.labels[idx])


# Cached datasets
_TRAIN_DS_CACHE: _InMemoryCIFAR10 | None = None
_TEST_DS_CACHE:  _InMemoryCIFAR10 | None = None
_TRAIN_LABELS_CACHE: np.ndarray | None = None
_ROOT_USED: Path | None = None


def _torchvision_to_tensor(ds: CIFAR10) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Convert a torchvision CIFAR10 (PIL backend) into a uint8 (N, 3, 32, 32)
    tensor + int64 label array, in one shot.
    """
    # ds.data: numpy uint8 (N, 32, 32, 3) ; ds.targets: list[int]
    imgs = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()    # NHWC → NCHW
    labels = np.asarray(ds.targets, dtype=np.int64)
    return imgs, labels


def _get_train_test(root: Path | None) -> Tuple[_InMemoryCIFAR10, _InMemoryCIFAR10, np.ndarray]:
    """Return cached train + test datasets; first call decodes everything."""
    global _TRAIN_DS_CACHE, _TEST_DS_CACHE, _TRAIN_LABELS_CACHE, _ROOT_USED

    if root is None:
        root = _discover_root(allow_download=True)

    if _TRAIN_DS_CACHE is None or _ROOT_USED != root:
        print(f"[CIFAR10] First-use decode from {root}/  (one-time, ~5 s) ...")
        # torchvision will auto-download if missing
        train_tv = CIFAR10(root=str(root), train=True,  download=True)
        test_tv  = CIFAR10(root=str(root), train=False, download=True)

        train_imgs, train_labels = _torchvision_to_tensor(train_tv)
        test_imgs,  test_labels  = _torchvision_to_tensor(test_tv)

        _TRAIN_DS_CACHE = _InMemoryCIFAR10(train_imgs, train_labels)
        _TEST_DS_CACHE  = _InMemoryCIFAR10(test_imgs,  test_labels)
        _TRAIN_LABELS_CACHE = train_labels
        _ROOT_USED = root

        mb = (train_imgs.numel() + test_imgs.numel()) / (1024 * 1024)
        print(f"[CIFAR10] Loaded {train_imgs.shape[0]} train + {test_imgs.shape[0]} "
              f"test samples ({mb:.1f} MB resident).")

    return _TRAIN_DS_CACHE, _TEST_DS_CACHE, _TRAIN_LABELS_CACHE


# Dirichlet partition + DataLoader builder
def build_cifar10_loaders(
    num_clients: int,
    batch_size: int,
    seed: int,
    dirichlet_alpha: float = 0.3,
    root: Path | None = None,
) -> Tuple[List[DataLoader], DataLoader, List[int]]:
    """
    Dirichlet(α) non-IID partition of CIFAR-10 training set across `num_clients`.

    Mirrors build_dhcd_loaders / build_har_loaders signature exactly so the
    simulator can dispatch on dataset name without other changes.

    Returns
    -------
    client_loaders        : list[DataLoader], one per client
    test_loader           : DataLoader over the full 10k test split, batch 256
    n_samples_per_client  : per-client training-sample count
    """
    train_ds, test_ds, train_labels = _get_train_test(root)
    np_rng = np.random.RandomState(seed)

    # Per-class Dirichlet split (Hsu et al. 2019), deterministic given seed.
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for cls in range(CIFAR10_N_CLASSES):
        cls_idx = np.where(train_labels == cls)[0]
        np_rng.shuffle(cls_idx)
        props = np_rng.dirichlet(np.ones(num_clients) * dirichlet_alpha)
        counts = (props * len(cls_idx)).astype(int)
        counts[-1] = len(cls_idx) - counts[:-1].sum()    # absorb rounding error
        start = 0
        for c, n in enumerate(counts):
            if n > 0:
                client_indices[c].extend(cls_idx[start:start + n].tolist())
                start += n

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
            num_workers=0,
            pin_memory=False,
        )
        client_loaders.append(loader)
        n_samples_per_client.append(len(idx))

    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    return client_loaders, test_loader, n_samples_per_client


# Smoke test
if __name__ == "__main__":
    print("CIFAR-10 loader smoke test")
    root = _discover_root(allow_download=True)
    print(f"  Root resolved to: {root}")
    loaders, test_loader, ns = build_cifar10_loaders(
        num_clients=10, batch_size=32, seed=42, dirichlet_alpha=0.3)
    print(f"  Built {len(loaders)} client loaders")
    print(f"  Per-client sample counts: {ns}  (sum={sum(ns)})")
    print(f"  Test loader: {len(test_loader.dataset)} samples")
    x, y = next(iter(loaders[0]))
    print(f"  Client 0 first batch: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}, "
          f"x.dtype={x.dtype}, y range=[{int(y.min())}, {int(y.max())}]")
    print("Smoke OK.")
