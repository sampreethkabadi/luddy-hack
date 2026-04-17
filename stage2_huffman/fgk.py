"""Adaptive Huffman coding -- FGK algorithm (Faller, Gallager, Knuth).

Owner: Sampreeth

Core idea:
- Both encoder and decoder maintain an identical Huffman tree that updates
  after every symbol. No preamble/frequency table is transmitted -- the tree
  is rebuilt in lockstep.
- A special NYT ("Not Yet Transmitted") leaf represents all unseen symbols.
  The first occurrence of a symbol is sent as NYT-code + raw symbol bits.
- After each symbol, weights are incremented and the tree is rebalanced while
  preserving the "sibling property": nodes listed in non-decreasing weight
  order have siblings adjacent in that list.

Public API (shape to confirm during impl):
- encode(text: str) -> bytes
- decode(payload: bytes) -> str

Use bitio.BitWriter / BitReader for the bit stream.

Invariants to test aggressively:
- Round-trip: decode(encode(x)) == x for all x including empty, 1-char,
  all-same-char, unicode, and long random strings.
- Sibling property holds after every update (add an internal assertion).

Not yet implemented.
"""
