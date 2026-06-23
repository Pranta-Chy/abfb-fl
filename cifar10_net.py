"""3-conv-block CNN for CIFAR-10 (~357k parameters)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10Net(nn.Module):
    """3-conv CNN for 10-class CIFAR-10 classification.

    Class attributes exposed for the baseline energy/MAC accounting.
    """

    N_CLASSES   = 10
    INPUT_CH    = 3
    INPUT_HW    = 32
    INPUT_SHAPE = (3, 32, 32)

    # Forward MAC budget per sample (matches cifar10net_macs below):
    #   conv1: 32×32×32×3×9      =   884,736
    #   conv2: 16×16×64×32×9     = 4,718,592
    #   conv3: 8×8×128×64×9      = 4,718,592
    #   fc1:   2048×128          =   262,144
    #   fc2:   128×10            =     1,280
    #   Total                    = 10,585,344 MACs / sample
    FORWARD_MACS_PER_SAMPLE = 10_585_344
    TOTAL_MACS_PER_STEP     = int(FORWARD_MACS_PER_SAMPLE * 3.5)   # +backward

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.3):
        super().__init__()
        # Block 1: 3→32, 32×32 → 16×16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        # Block 2: 32→64, 16×16 → 8×8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        # Block 3: 64→128, 8×8 → 4×4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        # Head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)     # 2048 → 128
        self.fc2 = nn.Linear(128, n_classes)       # 128 → 10

    def forward(self, x):
        # Accept (B, 3, 32, 32). Reject anything else.
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(
                f"CIFAR10Net expects (B, 3, 32, 32); got {tuple(x.shape)}")
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = x.flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def cifar10net_param_count() -> int:
    """Quick sanity check on parameter count."""
    m = CIFAR10Net()
    return sum(p.numel() for p in m.parameters())


def cifar10net_macs(batch_size: int = 1) -> int:
    """
    Forward-pass MAC count for a single input.
    Matches FORWARD_MACS_PER_SAMPLE; verified by hand below.
    """
    macs = (
        32 * 32 * 32 * 3 * 9         # conv1
        + 16 * 16 * 64 * 32 * 9      # conv2
        + 8 * 8 * 128 * 64 * 9       # conv3
        + 2048 * 128                 # fc1
        + 128 * 10                   # fc2
    )
    return macs * batch_size


if __name__ == "__main__":
    m = CIFAR10Net()
    n_params = cifar10net_param_count()
    macs = cifar10net_macs()
    print("CIFAR10Net:")
    print(f"  Params:            {n_params:,}")
    print(f"  Forward MACs:      {macs:,}  ({macs/1e6:.2f} M)")
    print(f"  FP32 weight size:  {n_params * 4 / 1024:.1f} KB "
          f"({n_params * 4 / 1024 / 1024:.2f} MB)")
    # Forward sanity check
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(f"  Forward shape:     {tuple(y.shape)}")
    assert y.shape == (2, 10)
    print("Sanity OK.")
