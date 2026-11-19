from pathlib import Path

import numpy as np

from budget_aware_search.serving.matrix_io import (
    read_float32_matrix,
    read_int64_matrix,
    write_float32_matrix,
    write_int64_matrix,
)


def test_float_matrix_round_trip(tmp_path: Path):
    matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "matrix.f32mat"
    write_float32_matrix(path, matrix)
    np.testing.assert_array_equal(read_float32_matrix(path), matrix)


def test_int64_matrix_round_trip(tmp_path: Path):
    matrix = np.arange(10, dtype=np.int64).reshape(2, 5)
    path = tmp_path / "matrix.i64mat"
    write_int64_matrix(path, matrix)
    np.testing.assert_array_equal(read_int64_matrix(path), matrix)
