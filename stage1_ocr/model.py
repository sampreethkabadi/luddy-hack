"""
model.py — CNN for Character Recognition (EMNIST)

Takes a 28x28 grayscale image of a single character and outputs which of the
62 EMNIST classes it belongs to (0-9, A-Z, a-z).

Architecture:
    [1×28×28] →Conv1→ [32×28×28] →Pool→ [32×14×14]
              →Conv2→ [64×14×14]  →Pool→ [64×7×7]
              →Conv3→ [128×7×7]
              →Flatten→ [6272]
              →FC1→ [256] →FC2→ [62]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OCRNet(nn.Module):
    """CNN for 62-class EMNIST character recognition."""

    def __init__(self, num_classes=62):
        super(OCRNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool         = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc   = nn.Dropout(0.5)

        self.fc1    = nn.Linear(128 * 7 * 7, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2    = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1: [B,1,28,28] → [B,32,14,14]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2: [B,32,14,14] → [B,64,7,7]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)

        # Block 3: [B,64,7,7] → [B,128,7,7]
        x = F.relu(self.bn3(self.conv3(x)))

        # Classifier
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(F.relu(self.bn_fc1(self.fc1(x))))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# EMNIST label mapping
# ---------------------------------------------------------------------------

EMNIST_LABELS = list(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

def index_to_char(idx: int) -> str:
    return EMNIST_LABELS[idx]

def char_to_index(char: str) -> int:
    return EMNIST_LABELS.index(char)


# ---------------------------------------------------------------------------
# Quick summary
# ---------------------------------------------------------------------------

def print_model_summary():
    model = OCRNet(num_classes=62)
    total = sum(p.numel() for p in model.parameters())
    print(f"OCRNet  —  {total:,} parameters  (~{total*4/1024/1024:.1f} MB float32)")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 28, 28))
    print(f"Input: (1,1,28,28)  Output: {tuple(out.shape)}")


if __name__ == "__main__":
    print_model_summary()
