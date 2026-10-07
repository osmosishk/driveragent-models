"""Run-length encode/decode binary segmentation masks for ZMQ transport.

Format: row-major over the flattened mask. Output bytes are a sequence of
records, each:
    count : uint16 little-endian   # 1..65535
    value : uint8                  # 0 or non-zero (treated as 1 on decode)

Runs longer than 65535 are split. For typical road scenes the compression
ratio is ~30-100x relative to raw 1280x720 (= 921600 bytes).
"""

import numpy as np

_MAX_RUN = 65535


def encode_rle(mask: np.ndarray) -> bytes:
    """mask: 2D uint8 array (0 / non-zero). Returns RLE byte string."""
    flat = np.ascontiguousarray(mask, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return b""
    # Find boundary indices where the value changes.
    diff = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], diff))
    ends = np.concatenate((diff, [flat.size]))
    lengths = ends - starts
    values = flat[starts]

    pieces = []
    for length, value in zip(lengths, values):
        # Normalize non-zero to 1 to keep the wire format compact.
        v = np.uint8(1 if value else 0)
        remaining = int(length)
        while remaining > _MAX_RUN:
            pieces.append(np.uint16(_MAX_RUN).tobytes())
            pieces.append(bytes((int(v),)))
            remaining -= _MAX_RUN
        if remaining > 0:
            pieces.append(np.uint16(remaining).tobytes())
            pieces.append(bytes((int(v),)))
    return b"".join(pieces)


def decode_rle(buf: bytes, h: int, w: int) -> np.ndarray:
    """Inverse of encode_rle. Returns uint8 [h, w] in {0, 1}."""
    expected = h * w
    if not buf:
        return np.zeros((h, w), dtype=np.uint8)
    raw = np.frombuffer(buf, dtype=np.uint8)
    # Each record is 3 bytes (uint16 LE count + uint8 value).
    if raw.size % 3 != 0:
        raise ValueError(f"RLE buffer length {raw.size} not multiple of 3")
    counts = raw.reshape(-1, 3)[:, :2].copy().view(np.uint16).reshape(-1)
    values = raw.reshape(-1, 3)[:, 2]
    out = np.repeat(values, counts.astype(np.int64))
    if out.size != expected:
        # Pad or truncate defensively; should not happen if encoder matches.
        if out.size < expected:
            out = np.concatenate([out, np.zeros(expected - out.size, dtype=np.uint8)])
        else:
            out = out[:expected]
    return out.reshape(h, w)
