"""CNN OCR model definitions.

Owner: Anuj

Three variants (see PLAN.md §3 Experiment Matrix):
  A. PureCNN_CTC        — conv stack → linear projection → CTC
  B. CRNN_CTC           — conv stack → BiLSTM → CTC  (recommended default)
  C. UNetDenoiserCRNN   — U-Net denoiser head + CRNN OCR head, dual loss

All models share the same CNN backbone (FeatureExtractor).

Input to OCR models:  (N, 1, 40, W) float32 tensors — line crops.
Output:               (T, N, num_classes) log-softmax, ready for CTCLoss.

Save / load weights:
    torch.save({'arch': 'crnn_ctc', 'state': model.state_dict(),
                'alphabet': alphabet.chars}, 'stage1_ocr/weights/best.pt')

    ckpt = torch.load('stage1_ocr/weights/best.pt')
    model = build_model(ckpt['arch'], num_classes=len(ckpt['alphabet']))
    model.load_state_dict(ckpt['state'])
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared CNN backbone
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class FeatureExtractor(nn.Module):
    """Shared CNN backbone for all three OCR variants.

    Input:  (N, 1, 40, W)
    Output: (T, N, 256)  — height fully collapsed; W//8 timesteps.

    Pooling schedule:
        block 1: MaxPool(2,2)  → (N, 32,  20, W//2)
        block 2: MaxPool(2,2)  → (N, 64,  10, W//4)
        block 3: MaxPool(2,2)  → (N, 128,  5, W//8)
        block 4: MaxPool(5,1)  → (N, 256,  1, W//8)  — height → 1
    """

    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(_ConvBnRelu(1, 32),   nn.MaxPool2d((2, 2)))
        self.block2 = nn.Sequential(_ConvBnRelu(32, 64),  nn.MaxPool2d((2, 2)))
        self.block3 = nn.Sequential(_ConvBnRelu(64, 128), nn.MaxPool2d((2, 2)))
        self.block4 = nn.Sequential(_ConvBnRelu(128, 256), nn.MaxPool2d((5, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)    # (N, 32,  20, W//2)
        x = self.block2(x)    # (N, 64,  10, W//4)
        x = self.block3(x)    # (N, 128,  5, W//8)
        x = self.block4(x)    # (N, 256,  1, W//8)
        x = x.squeeze(2)      # (N, 256, T)
        x = x.permute(2, 0, 1)  # (T, N, 256)
        return x


# ---------------------------------------------------------------------------
# Variant A — Pure CNN + CTC
# ---------------------------------------------------------------------------

class PureCNN_CTC(nn.Module):
    """Fastest baseline: conv features → linear head → CTC."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = FeatureExtractor()
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (T, N, 256)
        x = self.head(x)           # (T, N, C)
        return F.log_softmax(x, dim=2)


# ---------------------------------------------------------------------------
# Variant B — CRNN + CTC  (recommended default)
# ---------------------------------------------------------------------------

class CRNN_CTC(nn.Module):
    """Industry-standard line OCR: CNN features → BiLSTM → CTC."""

    def __init__(
        self,
        num_classes: int,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.features = FeatureExtractor()
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.head = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (T, N, 256)
        x, _ = self.rnn(x)        # (T, N, 512)
        x = self.head(x)           # (T, N, C)
        return F.log_softmax(x, dim=2)


# ---------------------------------------------------------------------------
# Variant C — U-Net denoiser + CRNN  (dual-loss)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNetDenoiser(nn.Module):
    """Lightweight 3-level U-Net for grayscale image denoising.

    Input / output: (N, 1, H, W) — full page images.
    Used as the reconstruction head in UNetDenoiserCRNN.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = _DoubleConv(1, 32)
        self.enc2 = _DoubleConv(32, 64)
        self.enc3 = _DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(128, 64)   # 64 skip + 64 up
        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(64, 32)    # 32 skip + 32 up
        self.out  = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


class UNetDenoiserCRNN(nn.Module):
    """Two-headed model: U-Net denoiser + CRNN OCR.

    Training loss = alpha * MSE(denoised, clean) + beta * CTC(text)

    forward(noisy_page)       -> denoised_page   (for reconstruction loss)
    forward_ocr(line_crops)   -> log_probs        (for CTC loss)

    During inference the service:
      1. Calls forward(noisy_page) to get denoised_page.
      2. Runs line segmentation on denoised_page.
      3. Calls forward_ocr(line_crops) to get text.
    """

    def __init__(
        self,
        num_classes: int,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.denoiser = UNetDenoiser()
        self.crnn     = CRNN_CTC(num_classes, rnn_hidden, rnn_layers)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Denoising pass — returns reconstructed page (N, 1, H, W)."""
        return self.denoiser(noisy)

    def forward_ocr(self, line_crops: torch.Tensor) -> torch.Tensor:
        """OCR pass on pre-extracted line crops — returns (T, N, C) log-probs."""
        return self.crnn(line_crops)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "pure_cnn_ctc":        PureCNN_CTC,
    "crnn_ctc":            CRNN_CTC,
    "unet_denoiser_crnn":  UNetDenoiserCRNN,
}


def build_model(arch: str, num_classes: int, **kwargs) -> nn.Module:
    """Instantiate a model by architecture name.

    Args:
        arch:        One of 'pure_cnn_ctc', 'crnn_ctc', 'unet_denoiser_crnn'.
        num_classes: Alphabet size including the CTC blank token.
        **kwargs:    Forwarded to the model constructor (e.g. rnn_hidden=512).
    """
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown arch {arch!r}. Choose from {list(_REGISTRY)}")
    return _REGISTRY[arch](num_classes=num_classes, **kwargs)
