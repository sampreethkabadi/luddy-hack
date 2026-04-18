"""NoisyOffice dataset loader.

Owner: Sampreeth

Filename schema (inside SimulatedNoisyOffice/):
    noisy_images_grayscale/   Font{size}{type}{emph}_Noise{D}_{split}.png
    clean_images_grayscale/   Font{size}{type}{emph}_Clean_{split}.png

  size  : f | n | L
  type  : t | s | r
  emph  : e | m
  D     : f (folded) | w (wrinkled) | c (coffee) | p (footprint)
  split : TR | VA | TE

Yields named tuples:
    NoisySample(noisy, clean, label, lines, noise_type, font, split)

where noisy / clean are float32 tensors of shape (1, H, W), values in [0, 1].

Usage:
    ds = NoisyOfficeDataset("SimulatedNoisyOffice", split="TR", labels_path="data/labels.json")
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    for batch in loader:
        images   = batch["noisy"]       # (B, 1, H, W)
        targets  = batch["label"]       # list[str] of length B
        noise    = batch["noise_type"]  # list[str]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOISE_TYPES = ("f", "w", "c", "p")  # folded, wrinkled, coffee, footprint
SPLITS = ("TR", "VA", "TE")

_NOISY_RE = re.compile(
    r"^(?P<font>[A-Za-z0-9]+)_Noise(?P<noise>[a-zA-Z])_(?P<split>TR|VA|TE)\.png$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NoisySample:
    noisy: torch.Tensor       # (1, H, W) float32
    clean: torch.Tensor       # (1, H, W) float32
    label: str                # full transcription from labels.json
    lines: list[str]          # per-line transcriptions
    noise_type: str           # "f" | "w" | "c" | "p"
    font: str                 # e.g. "FontLre"
    split: str                # "TR" | "VA" | "TE"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NoisyOfficeDataset(Dataset):
    """PyTorch Dataset for SimulatedNoisyOffice.

    Args:
        root: Path to the SimulatedNoisyOffice directory.
        split: One of "TR", "VA", "TE", or None for all splits.
        noise_types: Subset of ("f","w","c","p") to include. None = all.
        labels_path: Path to data/labels.json.
        transform: Optional torchvision transform applied to *both* tensors.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["TR", "VA", "TE"] | None = None,
        noise_types: tuple[str, ...] | None = None,
        labels_path: str | Path = "data/labels.json",
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.noisy_dir = self.root / "noisy_images_grayscale"
        self.clean_dir = self.root / "clean_images_grayscale"

        if not self.noisy_dir.exists():
            raise FileNotFoundError(f"Noisy image dir not found: {self.noisy_dir}")
        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean image dir not found: {self.clean_dir}")

        self.labels: dict = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self.transform = transform
        self._to_tensor = transforms.ToTensor()

        allowed_noise = set(noise_types) if noise_types else set(NOISE_TYPES)

        self.samples: list[dict] = []
        for png in sorted(self.noisy_dir.glob("*.png")):
            m = _NOISY_RE.match(png.name)
            if m is None:
                continue
            font = m.group("font")
            noise = m.group("noise").lower()
            sp = m.group("split").upper()

            if split is not None and sp != split:
                continue
            if noise not in allowed_noise:
                continue

            label_key = f"{font}_{sp}"
            if label_key not in self.labels:
                continue

            clean_name = f"{font}_Clean_{sp}.png"
            clean_path = self.clean_dir / clean_name
            if not clean_path.exists():
                continue

            self.samples.append(
                dict(
                    noisy_path=png,
                    clean_path=clean_path,
                    label=self.labels[label_key]["full"],
                    lines=self.labels[label_key]["lines"],
                    noise_type=noise,
                    font=font,
                    split=sp,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        noisy_img = Image.open(s["noisy_path"]).convert("L")
        clean_img = Image.open(s["clean_path"]).convert("L")

        noisy_t: torch.Tensor = self._to_tensor(noisy_img)   # (1, H, W)
        clean_t: torch.Tensor = self._to_tensor(clean_img)

        if self.transform:
            noisy_t = self.transform(noisy_t)
            clean_t = self.transform(clean_t)

        return {
            "noisy": noisy_t,
            "clean": clean_t,
            "label": s["label"],
            "lines": s["lines"],
            "noise_type": s["noise_type"],
            "font": s["font"],
            "split": s["split"],
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def filter_by_noise(self, noise_type: str) -> "NoisyOfficeDataset":
        """Return a view filtered to one noise type (no disk re-scan)."""
        subset = NoisyOfficeDataset.__new__(NoisyOfficeDataset)
        subset.root = self.root
        subset.noisy_dir = self.noisy_dir
        subset.clean_dir = self.clean_dir
        subset.labels = self.labels
        subset.transform = self.transform
        subset._to_tensor = self._to_tensor
        subset.samples = [s for s in self.samples if s["noise_type"] == noise_type]
        return subset

    def noise_breakdown(self) -> dict[str, int]:
        """Return count of samples per noise type."""
        counts: dict[str, int] = {n: 0 for n in NOISE_TYPES}
        for s in self.samples:
            counts[s["noise_type"]] += 1
        return counts


# ---------------------------------------------------------------------------
# Filename parser (standalone utility used by other modules)
# ---------------------------------------------------------------------------

def parse_noisy_filename(fname: str) -> tuple[str, str, str] | None:
    """Parse a noisy image filename → (font, noise_type, split) or None."""
    m = _NOISY_RE.match(fname)
    if m is None:
        return None
    return m.group("font"), m.group("noise").lower(), m.group("split").upper()
