"""CTC decoding utilities and alphabet management.

Owner: Anuj

Public API:
    build_alphabet(labels_path)  -> Alphabet
    Alphabet.encode(text)        -> list[int]
    Alphabet.decode(indices)     -> str
    greedy_decode(log_probs, alphabet) -> (str, confidences)
    greedy_decode_batch(log_probs, alphabet) -> list[str]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# CTC blank token is always index 0.
BLANK_IDX = 0
BLANK_CHAR = "\x00"   # internal sentinel — never appears in decoded output


# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------

@dataclass
class Alphabet:
    """Bidirectional character <-> index mapping.

    Index 0 is always the CTC blank token.
    Printable characters start at index 1.
    """
    chars: list[str]
    _c2i: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._c2i = {c: i for i, c in enumerate(self.chars)}

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Map each character to its index. Unknown chars map to blank."""
        return [self._c2i.get(c, BLANK_IDX) for c in text]

    def decode(self, indices: Sequence[int]) -> str:
        """Map indices to characters (does NOT collapse CTC repeats)."""
        return "".join(
            self.chars[i] for i in indices if 0 <= i < len(self.chars)
        )

    def __contains__(self, ch: str) -> bool:
        return ch in self._c2i

    def __repr__(self) -> str:
        return f"Alphabet(size={self.size}, chars={(''.join(self.chars[1:]))!r})"


# ---------------------------------------------------------------------------
# Alphabet construction
# ---------------------------------------------------------------------------

def build_alphabet(labels_path: str | Path = "data/labels.json") -> Alphabet:
    """Derive the character set from all text in labels.json.

    Blank token is placed at index 0; all other chars are sorted
    lexicographically for reproducibility across runs.
    """
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))

    char_set: set[str] = set()
    for entry in labels.values():
        for line in entry.get("lines", []):
            char_set.update(line)
        char_set.update(entry.get("full", ""))

    sorted_chars = [BLANK_CHAR] + sorted(char_set)
    return Alphabet(chars=sorted_chars)


# ---------------------------------------------------------------------------
# CTC greedy decoder
# ---------------------------------------------------------------------------

def greedy_decode(log_probs, alphabet: Alphabet) -> tuple[str, list[float]]:
    """Decode one sequence with greedy (best-path) CTC.

    Args:
        log_probs: (T, num_classes) — log-softmax values.
                   Accepts torch.Tensor, numpy array, or nested list.
        alphabet:  Alphabet matching the model output dimension.

    Returns:
        (text, confidences) where confidences[t] = exp(max log-prob) at step t.
    """
    try:
        import torch
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().tolist()
    except ImportError:
        pass

    best_indices: list[int] = []
    confidences: list[float] = []

    for timestep in log_probs:
        best_idx = max(range(len(timestep)), key=lambda i: timestep[i])
        best_indices.append(best_idx)
        confidences.append(math.exp(timestep[best_idx]))

    # Collapse consecutive duplicates, then strip blanks.
    collapsed: list[int] = []
    prev = None
    for idx in best_indices:
        if idx != prev:
            collapsed.append(idx)
        prev = idx

    text = "".join(
        alphabet.chars[i]
        for i in collapsed
        if i != BLANK_IDX and 0 <= i < alphabet.size
    )
    return text, confidences


def greedy_decode_batch(
    log_probs,
    alphabet: Alphabet,
    lengths: Sequence[int] | None = None,
) -> list[str]:
    """Decode a batch of sequences.

    Args:
        log_probs: (T, N, C) tensor or nested list.
        alphabet:  Shared alphabet.
        lengths:   Optional per-sample valid timestep counts (N,).

    Returns:
        List of N decoded strings.
    """
    try:
        import torch
        if isinstance(log_probs, torch.Tensor):
            T, N, _ = log_probs.shape
            results = []
            for n in range(N):
                t = lengths[n] if lengths is not None else T
                text, _ = greedy_decode(log_probs[:t, n, :], alphabet)
                results.append(text)
            return results
    except (ImportError, Exception):
        pass

    # Pure-Python fallback for nested lists.
    T = len(log_probs)
    N = len(log_probs[0]) if T > 0 else 0
    results = []
    for n in range(N):
        t = lengths[n] if lengths is not None else T
        seq = [log_probs[t_][n] for t_ in range(t)]
        text, _ = greedy_decode(seq, alphabet)
        results.append(text)
    return results
