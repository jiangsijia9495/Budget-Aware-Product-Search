#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
struct Matrix {
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  std::vector<T> values;

  const T* row_ptr(std::uint64_t row) const {
    return values.data() + row * cols;
  }

  T* row_ptr(std::uint64_t row) {
    return values.data() + row * cols;
  }
};

template <typename T>
Matrix<T> read_matrix(const std::string& path, const std::array<char, 4>& expected_magic) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open matrix file: " + path);
  }

  std::array<char, 4> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != expected_magic) {
    throw std::runtime_error("Unexpected matrix magic in: " + path);
  }

  Matrix<T> matrix;
  input.read(reinterpret_cast<char*>(&matrix.rows), sizeof(matrix.rows));
  input.read(reinterpret_cast<char*>(&matrix.cols), sizeof(matrix.cols));
  if (!input || matrix.rows == 0 || matrix.cols == 0) {
    throw std::runtime_error("Invalid matrix header in: " + path);
  }

  const std::uint64_t count = matrix.rows * matrix.cols;
  matrix.values.resize(static_cast<std::size_t>(count));
  input.read(
      reinterpret_cast<char*>(matrix.values.data()),
      static_cast<std::streamsize>(count * sizeof(T)));
  if (!input) {
    throw std::runtime_error("Matrix payload is truncated: " + path);
  }
  return matrix;
}

inline Matrix<float> read_float32_matrix(const std::string& path) {
  return read_matrix<float>(path, {'F', '3', '2', 'M'});
}

inline Matrix<std::int64_t> read_int64_matrix(const std::string& path) {
  return read_matrix<std::int64_t>(path, {'I', '6', '4', 'M'});
}
