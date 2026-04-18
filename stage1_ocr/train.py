"""Training loop for the OCR model.

Owner: Anuj

Intended execution environment: Colab GPU.

Quick start (Colab):
    !git clone <repo> && cd luddy-hack
    !pip install -r requirements.txt
    from stage1_ocr.train import Config, train
    cfg = Config(dataset_root="SimulatedNoisyOffice", arch="crnn_ctc", epochs=30)
    train(cfg)

CLI:
    python -m stage1_ocr.train --arch crnn_ctc --epochs 30 --dataset-root SimulatedNoisyOffice
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from benchmarks.metrics import cer as compute_cer
from data.dataset import NoisyOfficeDataset
from data.lines import segment_lines
from stage1_ocr.ctc import Alphabet, build_alphabet, greedy_decode_batch
from stage1_ocr.model import build_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    arch: Literal["pure_cnn_ctc", "crnn_ctc", "unet_denoiser_crnn"] = "crnn_ctc"
    dataset_root: str = "SimulatedNoisyOffice"
    labels_path:  str = "data/labels.json"
    weights_dir:  str = "stage1_ocr/weights"
    line_height:  int = 40
    batch_size:   int = 32
    epochs:       int = 30
    lr:           float = 3e-4
    rnn_hidden:   int = 256
    rnn_layers:   int = 2
    alpha:        float = 0.5   # MSE weight for UNetDenoiserCRNN
    beta:         float = 1.0   # CTC weight for UNetDenoiserCRNN
    seed:         int = 42
    noise_types:  tuple = ("f", "w", "c", "p")
    augment:      bool = True


# ---------------------------------------------------------------------------
# Line-level dataset  (wraps NoisyOfficeDataset, yields one line per sample)
# ---------------------------------------------------------------------------

class LineDataset(Dataset):
    """Each __getitem__ returns one text-line crop and its label.

    Segments page images lazily on first access and caches the result so
    repeated epochs don't re-run the segmenter on every call.
    """

    def __init__(
        self,
        page_ds: NoisyOfficeDataset,
        line_height: int = 40,
        augment: bool = False,
    ) -> None:
        self.page_ds    = page_ds
        self.line_height = line_height

        aug_list = []
        if augment:
            aug_list += [
                transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
            ]
        self.augment_fn = transforms.Compose(aug_list) if aug_list else None
        self.to_tensor  = transforms.ToTensor()

        # Pre-index: (page_idx, line_idx_within_page)
        self._index: list[tuple[int, int]] = []
        self._cache: dict[int, list] = {}   # page_idx -> list of (tensor, text, noise)
        self._build_index()

    def _build_index(self) -> None:
        print(f"  Indexing {len(self.page_ds)} page images ...")
        for page_idx in range(len(self.page_ds)):
            sample   = self.page_ds[page_idx]
            label_lines: list[str] = sample["lines"]
            noisy_pil = transforms.ToPILImage()(sample["noisy"])
            seg_crops = segment_lines(noisy_pil, target_height=self.line_height)

            # Align segmented crops with label lines.
            n_pairs = min(len(seg_crops), len(label_lines))
            if n_pairs == 0:
                continue

            page_lines = []
            for li in range(n_pairs):
                tensor = self.to_tensor(seg_crops[li])   # (1, H, W)
                page_lines.append((tensor, label_lines[li], sample["noise_type"]))

            self._cache[page_idx] = page_lines
            for li in range(n_pairs):
                self._index.append((page_idx, li))

        print(f"  {len(self._index)} line samples indexed.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        page_idx, line_idx = self._index[idx]
        tensor, text, noise_type = self._cache[page_idx][line_idx]
        if self.augment_fn is not None:
            tensor = self.augment_fn(tensor)
        return {"image": tensor, "text": text, "noise_type": noise_type}


# ---------------------------------------------------------------------------
# Collate: pad variable-width crops to the max width in the batch
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    max_w = max(s["image"].shape[2] for s in batch)
    images = torch.stack([
        torch.nn.functional.pad(s["image"], (0, max_w - s["image"].shape[2]))
        for s in batch
    ])
    return {
        "image":      images,
        "text":       [s["text"] for s in batch],
        "noise_type": [s["noise_type"] for s in batch],
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _encode_targets(texts: list[str], alphabet: Alphabet):
    """Encode a batch of label strings for CTCLoss.

    Returns:
        targets:        1-D LongTensor (concatenated encoded chars)
        target_lengths: 1-D LongTensor (length of each label)
    """
    encoded = [torch.tensor(alphabet.encode(t), dtype=torch.long) for t in texts]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    targets = torch.cat(encoded)
    return targets, target_lengths


def _run_ocr_forward(model, images: torch.Tensor, arch: str):
    """Forward pass that handles both plain OCR and UNetDenoiserCRNN."""
    if arch == "unet_denoiser_crnn":
        return model.forward_ocr(images), None   # denoiser path skipped for lines
    return model(images), None


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc_loss: nn.CTCLoss,
    alphabet: Alphabet,
    device: torch.device,
    arch: str,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)           # (N, 1, 40, W)
        texts  = batch["text"]

        log_probs, _ = _run_ocr_forward(model, images, arch)  # (T, N, C)
        T, N, _ = log_probs.shape
        input_lengths = torch.full((N,), T, dtype=torch.long)

        targets, target_lengths = _encode_targets(texts, alphabet)
        targets = targets.to(device)

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    alphabet: Alphabet,
    device: torch.device,
    arch: str,
) -> dict:
    """Evaluate and return per-noise-type CER + overall CER."""
    model.eval()
    buckets: dict[str, list[float]] = {"f": [], "w": [], "c": [], "p": []}

    for batch in loader:
        images     = batch["image"].to(device)
        texts      = batch["text"]
        noise_types = batch["noise_type"]

        log_probs, _ = _run_ocr_forward(model, images, arch)
        hypotheses   = greedy_decode_batch(log_probs, alphabet)

        for hyp, ref, noise in zip(hypotheses, texts, noise_types):
            if noise in buckets:
                buckets[noise].append(compute_cer(ref, hyp))

    result: dict = {}
    all_cers: list[float] = []
    for noise, cers in buckets.items():
        if cers:
            mean = sum(cers) / len(cers)
            result[noise] = {"mean_cer": round(mean, 4), "n": len(cers)}
            all_cers.extend(cers)
        else:
            result[noise] = {"mean_cer": None, "n": 0}

    result["overall"] = {
        "mean_cer": round(sum(all_cers) / len(all_cers), 4) if all_cers else None,
        "n": len(all_cers),
    }
    return result


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(cfg: Config) -> nn.Module:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_dir = Path(cfg.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # -- Alphabet --
    alphabet = build_alphabet(cfg.labels_path)
    print(f"Alphabet: {alphabet.size} classes")

    # -- Datasets --
    print("Building TR dataset ...")
    tr_page_ds = NoisyOfficeDataset(
        cfg.dataset_root, split="TR",
        noise_types=cfg.noise_types,
        labels_path=cfg.labels_path,
    )
    print("Building VA dataset ...")
    va_page_ds = NoisyOfficeDataset(
        cfg.dataset_root, split="VA",
        noise_types=cfg.noise_types,
        labels_path=cfg.labels_path,
    )

    tr_ds = LineDataset(tr_page_ds, line_height=cfg.line_height, augment=cfg.augment)
    va_ds = LineDataset(va_page_ds, line_height=cfg.line_height, augment=False)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # -- Model --
    model = build_model(
        cfg.arch,
        num_classes=alphabet.size,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
    ).to(device)
    print(f"Model: {cfg.arch}  params={sum(p.numel() for p in model.parameters()):,}")

    # -- Loss / Optimizer / Scheduler --
    ctc_loss  = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_cer  = float("inf")
    best_path = weights_dir / "best.pt"

    # -- Training loop --
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, tr_loader, optimizer, ctc_loss,
                                 alphabet, device, cfg.arch)
        cer_report = eval_epoch(model, va_loader, alphabet, device, cfg.arch)

        overall_cer = cer_report["overall"]["mean_cer"] or 1.0
        scheduler.step(overall_cer)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"loss={train_loss:.4f}  "
            f"val_CER overall={overall_cer:.4f}  "
            f"f={cer_report['f']['mean_cer']}  "
            f"w={cer_report['w']['mean_cer']}  "
            f"c={cer_report['c']['mean_cer']}  "
            f"p={cer_report['p']['mean_cer']}  "
            f"({elapsed:.1f}s)"
        )

        if overall_cer < best_cer:
            best_cer = overall_cer
            torch.save(
                {
                    "arch":     cfg.arch,
                    "state":    model.state_dict(),
                    "alphabet": alphabet.chars,
                    "epoch":    epoch,
                    "cer":      best_cer,
                    "config":   cfg.__dict__,
                },
                best_path,
            )
            print(f"  ✓ Saved best model  CER={best_cer:.4f}  → {best_path}")

    print(f"\nTraining complete. Best val CER: {best_cer:.4f}")
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train OCR model on NoisyOffice dataset")
    p.add_argument("--arch", default="crnn_ctc",
                   choices=["pure_cnn_ctc", "crnn_ctc", "unet_denoiser_crnn"])
    p.add_argument("--dataset-root", default="SimulatedNoisyOffice")
    p.add_argument("--labels-path",  default="data/labels.json")
    p.add_argument("--weights-dir",  default="stage1_ocr/weights")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--rnn-hidden",   type=int,   default=256)
    p.add_argument("--rnn-layers",   type=int,   default=2)
    p.add_argument("--no-augment",   action="store_true")
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()
    return Config(
        arch=args.arch,
        dataset_root=args.dataset_root,
        labels_path=args.labels_path,
        weights_dir=args.weights_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        augment=not args.no_augment,
        seed=args.seed,
    )


if __name__ == "__main__":
    train(_parse_args())
