"""Bit-level I/O primitives for adaptive Huffman.

Owner: Sampreeth

Two classes:
- BitWriter: buffers bits, flushes bytes, writes a small header recording how
  many meaningful bits are in the final byte (so decoding stops cleanly).
- BitReader: consumes a byte stream and yields individual bits; stops at the
  meaningful-bit boundary recorded in the header.

Header format (decide during impl, document here):
- 1 byte: number of meaningful bits in the last data byte (1..8)
- followed by the payload bytes

Round-trip-testable independently of the Huffman code above.

Not yet implemented.
"""
