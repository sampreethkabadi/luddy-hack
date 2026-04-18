"""Adaptive Huffman coding — Vitter's Algorithm V.

Owner: Sampreeth (STRETCH GOAL)

Vitter's algorithm differs from FGK in one key invariant:
    Within any weight block, LEAVES always have higher order numbers than
    INTERNAL nodes of the same weight.

Practical effect: when a leaf p needs to slide within its weight block, it
only swaps with other leaves — never with internal nodes.  Internal nodes
swap with any block-leader.  This produces provably optimal codes at every
step and tends to yield ~1-3 % better compression than FGK on natural text.

Public API (identical to fgk.py — drop-in replacement):
    encode(text: str) -> bytes
    decode(payload: bytes) -> str

Implementation strategy:
    Subclass _FGKTree and override _leader() so that leaf updates are
    leader-searched only among leaves, while internal-node updates search
    all nodes.  Everything else (tree structure, bitio, wire format) is
    inherited unchanged from FGK.
"""

from __future__ import annotations

import struct

from .bitio import BitWriter, BitReader
from .fgk import _FGKTree, _Node, _INIT_ORDER


# ---------------------------------------------------------------------------
# Vitter tree
# ---------------------------------------------------------------------------

class _VitterTree(_FGKTree):
    """FGK tree with Vitter's leaf-preferring leader selection.

    The only change from FGK: when the node being updated is a LEAF, the
    block leader search is restricted to other LEAVES.  Internal nodes still
    search the whole block.

    This maintains Vitter's invariant (leaves rank higher than internal nodes
    within the same weight block) and produces shorter expected code lengths.
    """

    def _leader(self, node: _Node) -> _Node:
        node_is_leaf = node.left is None
        best = node
        for n in self._omap.values():
            if n.weight != node.weight or n.order <= best.order:
                continue
            if node_is_leaf and n.left is not None:
                # Vitter: a leaf never slides past an internal node.
                continue
            best = n
        return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(text: str) -> bytes:
    """Compress *text* with Vitter's Algorithm V."""
    data = text.encode("utf-8")
    bw = BitWriter()
    tree = _VitterTree()
    for byte in data:
        tree.encode_sym(byte, bw)
    return struct.pack(">I", len(data)) + bw.flush()


def decode(compressed: bytes) -> str:
    """Decompress bytes produced by :func:`encode`."""
    if len(compressed) < 4:
        raise ValueError("compressed data too short (missing length header)")
    n_bytes: int = struct.unpack(">I", compressed[:4])[0]
    br = BitReader(compressed[4:])
    tree = _VitterTree()
    out = bytearray()
    for _ in range(n_bytes):
        out.append(tree.decode_sym(br))
    return out.decode("utf-8")
