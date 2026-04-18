"""Adaptive Huffman coding — FGK algorithm (Faller, Gallager, Knuth).

Public API:
    encode(text: str) -> bytes
    decode(payload: bytes) -> str

Both encoder and decoder maintain an identical tree that grows in lockstep,
so no frequency table is transmitted.  A NYT ("Not Yet Transmitted") leaf
represents all unseen symbols; the first occurrence of a byte is sent as
NYT-code + 8 raw bits.

Sibling property invariant: nodes listed in non-decreasing weight order have
strictly increasing order numbers; siblings are adjacent in that listing.

Wire format: 4-byte big-endian byte count || BitWriter payload.
"""

from __future__ import annotations

import struct
from .bitio import BitWriter, BitReader

# NYT starts at the top of the order space.
# Each split consumes 2 order slots below the current NYT, so
# 2 * 256 + 1 = 513 slots are sufficient for a 256-symbol alphabet.
_INIT_ORDER = 512


class _Node:
    __slots__ = ("weight", "parent", "left", "right", "symbol", "order")

    def __init__(
        self,
        weight: int = 0,
        symbol: int | None = None,
        order: int = 0,
    ) -> None:
        self.weight = weight
        self.symbol = symbol   # byte value for leaves; None for internal nodes
        self.order = order
        self.parent: _Node | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None


class _FGKTree:
    def __init__(self) -> None:
        self.nyt = _Node(order=_INIT_ORDER)
        self.root = self.nyt
        self._sym: dict[int, _Node] = {}           # symbol -> leaf node
        self._omap: dict[int, _Node] = {_INIT_ORDER: self.nyt}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _code(self, node: _Node) -> list[int]:
        """Return the bit path from root to node (0=left, 1=right)."""
        bits: list[int] = []
        while node.parent is not None:
            bits.append(0 if node.parent.left is node else 1)
            node = node.parent
        return bits[::-1]

    def _leader(self, node: _Node) -> _Node:
        """Highest-order node in node's weight block (may be node itself)."""
        best = node
        for n in self._omap.values():
            if n.weight == node.weight and n.order > best.order:
                best = n
        return best

    def _swap(self, a: _Node, b: _Node) -> None:
        """Exchange a and b in the tree, including their order numbers."""
        # Swap order numbers (order is a property of the tree position).
        a.order, b.order = b.order, a.order
        self._omap[a.order] = a
        self._omap[b.order] = b

        # Swap tree positions.
        ap, bp = a.parent, b.parent
        if ap is bp:
            # Siblings — just flip left/right under the shared parent.
            ap.left, ap.right = ap.right, ap.left
        else:
            (ap.left if ap.left is a else None)  # noqa: unused
            if ap.left is a:
                ap.left = b
            else:
                ap.right = b
            if bp.left is b:
                bp.left = a
            else:
                bp.right = a
            a.parent, b.parent = bp, ap

    def _update(self, node: _Node) -> None:
        """Slide-and-increment from node up to the root."""
        while node is not None:
            leader = self._leader(node)
            if leader is not node and leader is not node.parent:
                self._swap(node, leader)
            node.weight += 1
            node = node.parent  # type: ignore[assignment]

    def _split_nyt(self, symbol: int) -> _Node:
        """Replace the current NYT leaf with an internal node.

        The internal node's left child becomes the new NYT, its right child
        becomes the new symbol leaf.  Returns the new leaf.
        """
        order = self.nyt.order
        internal = self.nyt          # reuse node object in-place
        internal.symbol = None

        new_nyt = _Node(order=order - 2)
        new_leaf = _Node(symbol=symbol, order=order - 1)

        internal.left = new_nyt
        internal.right = new_leaf
        new_nyt.parent = internal
        new_leaf.parent = internal

        self._omap[order - 2] = new_nyt
        self._omap[order - 1] = new_leaf

        self.nyt = new_nyt
        self._sym[symbol] = new_leaf
        return new_leaf

    # ------------------------------------------------------------------
    # Encode / decode one symbol
    # ------------------------------------------------------------------

    def encode_sym(self, symbol: int, bw: BitWriter) -> None:
        if symbol in self._sym:
            leaf = self._sym[symbol]
            for bit in self._code(leaf):
                bw.write_bit(bit)
            self._update(leaf)
        else:
            for bit in self._code(self.nyt):
                bw.write_bit(bit)
            bw.write_bits(symbol, 8)
            leaf = self._split_nyt(symbol)
            self._update(leaf)

    def decode_sym(self, br: BitReader) -> int:
        node = self.root
        while node.left is not None:
            node = node.left if next(br) == 0 else node.right  # type: ignore[assignment]

        if node is self.nyt:
            symbol = 0
            for _ in range(8):
                symbol = (symbol << 1) | next(br)
            leaf = self._split_nyt(symbol)
            self._update(leaf)
        else:
            symbol = node.symbol  # type: ignore[assignment]
            self._update(node)

        return symbol


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def encode(text: str) -> bytes:
    """Compress *text* and return the wire-format bytes."""
    data = text.encode("utf-8")
    bw = BitWriter()
    tree = _FGKTree()
    for byte in data:
        tree.encode_sym(byte, bw)
    return struct.pack(">I", len(data)) + bw.flush()


def decode(compressed: bytes) -> str:
    """Decompress bytes produced by :func:`encode`."""
    if len(compressed) < 4:
        raise ValueError("compressed data too short (missing length header)")
    n_bytes: int = struct.unpack(">I", compressed[:4])[0]
    br = BitReader(compressed[4:])
    tree = _FGKTree()
    out = bytearray()
    for _ in range(n_bytes):
        out.append(tree.decode_sym(br))
    return out.decode("utf-8")
