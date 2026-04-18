"""Bit-level I/O primitives for adaptive Huffman.

Header format (1 byte): number of meaningful bits in the last data byte (1–8).
  - Value 8 means the last byte is fully used (or there are no bytes at all).
  - Followed by the payload bytes.

Both classes are independent and round-trip testable on their own.
"""

from __future__ import annotations


class BitWriter:
    """Accumulates bits and packs them into bytes (MSB-first)."""

    def __init__(self) -> None:
        self._buf = 0    # bits being accumulated into the current byte
        self._nbits = 0  # how many bits are in _buf (0–7)
        self._data = bytearray()

    def write_bit(self, bit: int) -> None:
        self._buf = (self._buf << 1) | (bit & 1)
        self._nbits += 1
        if self._nbits == 8:
            self._data.append(self._buf)
            self._buf = 0
            self._nbits = 0

    def write_bits(self, value: int, n: int) -> None:
        for shift in range(n - 1, -1, -1):
            self.write_bit((value >> shift) & 1)

    def flush(self) -> bytes:
        """Finalise the stream and return header + payload bytes.

        May only be called once; mutates internal state.
        """
        if self._nbits > 0:
            # Pad the partial byte with zeros on the right.
            padded = self._buf << (8 - self._nbits)
            self._data.append(padded)
            meaningful = self._nbits
            self._nbits = 0
        else:
            # Either empty or the last byte was already complete.
            meaningful = 8

        return bytes([meaningful]) + bytes(self._data)


class BitReader:
    """Reads bits from a byte sequence produced by BitWriter.flush()."""

    def __init__(self, data: bytes) -> None:
        if len(data) < 1:
            raise ValueError("data must contain at least the 1-byte header")
        meaningful_last = data[0]
        payload = data[1:]
        n = len(payload)
        # Total number of valid bits across all payload bytes.
        self._total = (n - 1) * 8 + meaningful_last if n > 0 else 0
        self._payload = memoryview(payload)
        self._pos = 0

    def __iter__(self) -> BitReader:
        return self

    def __next__(self) -> int:
        if self._pos >= self._total:
            raise StopIteration
        byte_idx = self._pos >> 3
        bit_shift = 7 - (self._pos & 7)
        self._pos += 1
        return int((self._payload[byte_idx] >> bit_shift) & 1)

    def read_bit(self) -> int:
        return next(self)

    @property
    def exhausted(self) -> bool:
        return self._pos >= self._total
