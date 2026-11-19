from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_FLOAT_MAGIC = b"F32M"
_INT64_MAGIC = b"I64M"


def write_float32_matrix(path: str | Path, matrix: np.ndarray) -> None:
    _write_matrix(path, matrix, np.float32, _FLOAT_MAGIC)


def write_int64_matrix(path: str | Path, matrix: np.ndarray) -> None:
    _write_matrix(path, matrix, np.int64, _INT64_MAGIC)


def _write_matrix(path: str | Path, matrix: np.ndarray, dtype, magic: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(matrix, dtype=dtype)
    if contiguous.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {contiguous.shape}")
    with path.open("wb") as f:
        f.write(magic)
        f.write(struct.pack("<QQ", contiguous.shape[0], contiguous.shape[1]))
        f.write(contiguous.tobytes(order="C"))


def read_float32_matrix(path: str | Path) -> np.ndarray:
    return _read_matrix(path, np.float32, _FLOAT_MAGIC)


def read_int64_matrix(path: str | Path) -> np.ndarray:
    return _read_matrix(path, np.int64, _INT64_MAGIC)


def _read_matrix(path: str | Path, dtype, magic: bytes) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as f:
        actual_magic = f.read(4)
        if actual_magic != magic:
            raise ValueError(f"Unexpected matrix magic in {path}: {actual_magic!r}")
        rows, columns = struct.unpack("<QQ", f.read(16))
        data = np.frombuffer(f.read(), dtype=dtype)
    expected = rows * columns
    if data.size != expected:
        raise ValueError(f"Expected {expected} values in {path}, found {data.size}")
    return data.reshape(rows, columns)
