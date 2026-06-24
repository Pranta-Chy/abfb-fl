"""
================================================================================
  DHCDNet  -  CNN classifier for the Devanagari Handwritten Character Dataset.

  Input:  1 × 32 × 32 grayscale image
  Output: 46-class logits

  Architecture (3 conv blocks + FC head, ~530K params):
      Conv(1→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)     # 32 → 16
      Conv(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)    # 16 → 8
      Conv(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)   # 8 → 4
      Flatten → Dropout(0.3) → FC(2048→256) → ReLU → FC(256→46)

  Energy/MAC accounting compatible with baseline / baseline estimate_compute_macs.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DHCDNet(nn.Module):
    """CNN for 46-class Devanagari character classification.

    Class attributes for baseline's energy/MAC accounting.
    FORWARD_MACS_PER_SAMPLE matches dhcdnet_macs() below.
    """

    N_CLASSES = 46
    INPUT_CH  = 1
    INPUT_HW  = 32
    INPUT_SHAPE = (1, 32, 32)

    # MAC budget per forward (computed in dhcdnet_macs):
    # conv1: 32*32*32*1*9       = 294,912
    # conv2: 16*16*64*32*9      = 4,718,592
    # conv3: 8*8*128*64*9       = 4,718,592
    # fc1:   2048*256           = 524,288
    # fc2:   256*46             = 11,776
    # Total                     = 10,268,160 MACs / sample
    FORWARD_MACS_PER_SAMPLE = 10_268_160
    # baseline uses BACKWARD_FACTOR = 2.5 → total = forward × 3.5
    TOTAL_MACS_PER_STEP     = int(FORWARD_MACS_PER_SAMPLE * 3.5)

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.3):
        super().__init__()
        # Block 1: 1→32, 32×32 → 16×16
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        # Block 2: 32→64, 16×16 → 8×8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        # Block 3: 64→128, 8×8 → 4×4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        # Head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)     # 2048 → 256
        self.fc2 = nn.Linear(256, n_classes)       # 256 → 46

    def forward(self, x):
        # Accept (B, 32, 32) or (B, 1, 32, 32)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = x.flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def dhcdnet_param_count() -> int:
    """Quick sanity check on parameter count."""
    m = DHCDNet()
    return sum(p.numel() for p in m.parameters())


def dhcdnet_macs(batch_size: int = 1) -> int:
    """
    Forward-pass MAC count for a single input.

    Per layer:
      conv1: 32 × 32 × 32 × 1 × 9   = 294,912
      conv2: 16 × 16 × 64 × 32 × 9  = 4,718,592
      conv3: 8 × 8 × 128 × 64 × 9   = 4,718,592
      fc1:   2048 × 256             = 524,288
      fc2:   256 × 46               = 11,776
      Total ≈ 10,268,160 MACs / sample
    """
    macs = (
        32 * 32 * 32 * 1 * 9
        + 16 * 16 * 64 * 32 * 9
        + 8 * 8 * 128 * 64 * 9
        + 2048 * 256
        + 256 * 46
    )
    return macs * batch_size


if __name__ == "__main__":
    m = DHCDNet()
    n_params = dhcdnet_param_count()
    macs = dhcdnet_macs()
    print(f"DHCDNet:")
    print(f"  Params:        {n_params:,}")
    print(f"  Forward MACs:  {macs:,}  ({macs/1e6:.2f} M)")
    print(f"  FP32 weight size: {n_params * 4 / 1024:.1f} KB ({n_params * 4 / 1024 / 1024:.2f} MB)")
    # Forward sanity
    x = torch.randn(2, 1, 32, 32)
    y = m(x)
    print(f"  Forward shape:  {tuple(y.shape)}")
