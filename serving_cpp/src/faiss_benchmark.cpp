#include "matrix_io.h"

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Options {
  std::string catalog_path;
  std::string query_path;
  std::string index_type = "flat";
  int top_k = 100;
  int hnsw_m = 32;
  int ef_search = 128;
  int threads = 1;
  int warmup_queries = 50;
};

std::string require_value(int argc, char** argv, int& index) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("Missing value after ") + argv[index]);
  }
  return argv[++index];
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--catalog") {
      options.catalog_path = require_value(argc, argv, i);
    } else if (argument == "--queries") {
      options.query_path = require_value(argc, argv, i);
    } else if (argument == "--index") {
      options.index_type = require_value(argc, argv, i);
    } else if (argument == "--top-k") {
      options.top_k = std::stoi(require_value(argc, argv, i));
    } else if (argument == "--hnsw-m") {
      options.hnsw_m = std::stoi(require_value(argc, argv, i));
    } else if (argument == "--ef-search") {
      options.ef_search = std::stoi(require_value(argc, argv, i));
    } else if (argument == "--threads") {
      options.threads = std::stoi(require_value(argc, argv, i));
    } else if (argument == "--warmup") {
      options.warmup_queries = std::stoi(require_value(argc, argv, i));
    } else if (argument == "--help") {
      std::cout
          << "Usage: faiss_benchmark --catalog catalog.f32mat --queries queries.f32mat "
          << "[--index flat|hnsw] [--top-k 100] [--threads 1]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + argument);
    }
  }
  if (options.catalog_path.empty() || options.query_path.empty()) {
    throw std::runtime_error("--catalog and --queries are required");
  }
  if (options.top_k <= 0 || options.threads <= 0) {
    throw std::runtime_error("top-k and threads must be positive");
  }
  return options;
}

double percentile(const std::vector<double>& sorted_values, double probability) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  const double raw_index = probability * static_cast<double>(sorted_values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(raw_index);
  const std::size_t upper = std::min(lower + 1, sorted_values.size() - 1);
  const double fraction = raw_index - static_cast<double>(lower);
  return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
#ifdef _OPENMP
    omp_set_num_threads(options.threads);
#endif

    const Matrix<float> catalog = read_float32_matrix(options.catalog_path);
    const Matrix<float> queries = read_float32_matrix(options.query_path);
    if (catalog.cols != queries.cols) {
      throw std::runtime_error("Catalog and query dimensions do not match");
    }
    const int dimension = static_cast<int>(catalog.cols);
    const int top_k = std::min<int>(options.top_k, static_cast<int>(catalog.rows));

    std::unique_ptr<faiss::Index> index;
    if (options.index_type == "flat") {
      index = std::make_unique<faiss::IndexFlatIP>(dimension);
    } else if (options.index_type == "hnsw") {
      auto hnsw = std::make_unique<faiss::IndexHNSWFlat>(
          dimension, options.hnsw_m, faiss::METRIC_INNER_PRODUCT);
      hnsw->hnsw.efSearch = options.ef_search;
      index = std::move(hnsw);
    } else {
      throw std::runtime_error("Unsupported index type: " + options.index_type);
    }

    const auto build_start = std::chrono::steady_clock::now();
    index->add(static_cast<faiss::idx_t>(catalog.rows), catalog.values.data());
    const auto build_end = std::chrono::steady_clock::now();
    const double build_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start).count();

    std::vector<float> distances(static_cast<std::size_t>(top_k));
    std::vector<faiss::idx_t> labels(static_cast<std::size_t>(top_k));
    const std::uint64_t warmup = std::min<std::uint64_t>(
        queries.rows, static_cast<std::uint64_t>(std::max(options.warmup_queries, 0)));
    for (std::uint64_t row = 0; row < warmup; ++row) {
      index->search(1, queries.row_ptr(row), top_k, distances.data(), labels.data());
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(queries.rows));
    const auto all_start = std::chrono::steady_clock::now();
    for (std::uint64_t row = 0; row < queries.rows; ++row) {
      const auto start = std::chrono::steady_clock::now();
      index->search(1, queries.row_ptr(row), top_k, distances.data(), labels.data());
      const auto end = std::chrono::steady_clock::now();
      latencies_ms.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
    }
    const auto all_end = std::chrono::steady_clock::now();
    const double total_seconds =
        std::chrono::duration<double>(all_end - all_start).count();
    std::sort(latencies_ms.begin(), latencies_ms.end());

    const double qps = static_cast<double>(queries.rows) / std::max(total_seconds, 1e-12);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "{\n"
              << "  \"index\": \"" << options.index_type << "\",\n"
              << "  \"catalog_rows\": " << catalog.rows << ",\n"
              << "  \"query_rows\": " << queries.rows << ",\n"
              << "  \"dimension\": " << dimension << ",\n"
              << "  \"top_k\": " << top_k << ",\n"
              << "  \"threads\": " << options.threads << ",\n"
              << "  \"build_ms\": " << build_ms << ",\n"
              << "  \"p50_ms\": " << percentile(latencies_ms, 0.50) << ",\n"
              << "  \"p95_ms\": " << percentile(latencies_ms, 0.95) << ",\n"
              << "  \"p99_ms\": " << percentile(latencies_ms, 0.99) << ",\n"
              << "  \"qps_sequential\": " << qps << "\n"
              << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
