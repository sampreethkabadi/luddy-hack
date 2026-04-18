from __future__ import annotations

import struct
from .bitio import BitWriter, BitReader
from .fgk import _FGKTree, _Node, _INIT_ORDER


class _VitterTree(_FGKTree):
    # Vitter's invariant: within a weight block, leaves rank above internal nodes.
    # A leaf only slides past other leaves, never past an internal node.
    def _leader(self, node: _Node) -> _Node:
        node_is_leaf = node.left is None
        best = node
        for n in self._omap.values():
            if n.weight != node.weight or n.order <= best.order:
                continue
            if node_is_leaf and n.left is not None:
                continue
            best = n
        return best


def encode(text: str) -> bytes:
    data = text.encode("utf-8")
    bw = BitWriter()
    tree = _VitterTree()
    for byte in data:
        tree.encode_sym(byte, bw)
    return struct.pack(">I", len(data)) + bw.flush()


def decode(compressed: bytes) -> str:
    if len(compressed) < 4:
        raise ValueError("compressed data too short")
    n_bytes: int = struct.unpack(">I", compressed[:4])[0]
    br = BitReader(compressed[4:])
    tree = _VitterTree()
    out = bytearray()
    for _ in range(n_bytes):
        out.append(tree.decode_sym(br))
    return out.decode("utf-8")
