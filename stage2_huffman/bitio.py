from __future__ import annotations


class BitWriter:
    def __init__(self) -> None:
        self._buf = 0
        self._nbits = 0
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
        if self._nbits > 0:
            padded = self._buf << (8 - self._nbits)
            self._data.append(padded)
            meaningful = self._nbits
            self._nbits = 0
        else:
            meaningful = 8
        return bytes([meaningful]) + bytes(self._data)


class BitReader:
    def __init__(self, data: bytes) -> None:
        if len(data) < 1:
            raise ValueError("data must contain at least the 1-byte header")
        meaningful_last = data[0]
        payload = data[1:]
        n = len(payload)
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
