#include "matrix_io.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string model_path;
  std::string input_dir;
  int intra_op_threads = 1;
  int warmup_rows = 50;
  int max_rows = -1;
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
    if (argument == "--model") {
      options.model_path = require_value(argc, argv, i);
    } else if (argument == "--input-dir") {
      options.input_dir = require_value(argc, argv, i);
    } else if (argument == "--threads") {
      options.intra_op_threads = std::stoi(require_value(argc, argv, i));
    } else if (argument == "--warmup") {
      options.warmup_rows = std::stoi(require_value(argc, argv, i));
    } else if (argument == "--max-rows") {
      options.max_rows = std::stoi(require_value(argc, argv, i));
    } else if (argument == "--help") {
      std::cout << "Usage: onnx_benchmark --model model.onnx --input-dir DIR "
                   "[--threads 1] [--max-rows N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + argument);
    }
  }
  if (options.model_path.empty() || options.input_dir.empty()) {
    throw std::runtime_error("--model and --input-dir are required");
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
    const std::filesystem::path input_dir(options.input_dir);

    std::map<std::string, Matrix<std::int64_t>> inputs;
    for (const std::string& name : {"input_ids", "attention_mask", "token_type_ids"}) {
      const auto path = input_dir / (name + ".i64mat");
      if (std::filesystem::exists(path)) {
        inputs.emplace(name, read_int64_matrix(path.string()));
      }
    }
    if (inputs.empty()) {
      throw std::runtime_error("No *.i64mat input files found in " + options.input_dir);
    }
    const std::uint64_t rows = inputs.begin()->second.rows;
    const std::uint64_t sequence_length = inputs.begin()->second.cols;
    for (const auto& [name, matrix] : inputs) {
      if (matrix.rows != rows || matrix.cols != sequence_length) {
        throw std::runtime_error("Input matrix dimensions disagree for " + name);
      }
    }

    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "budget-aware-search");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(options.intra_op_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(environment, options.model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_name_storage;
    std::vector<const char*> input_names;
    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
      auto allocated_name = session.GetInputNameAllocated(i, allocator);
      input_name_storage.emplace_back(allocated_name.get());
    }
    for (const auto& name : input_name_storage) {
      if (!inputs.count(name)) {
        throw std::runtime_error("Missing token matrix for ONNX input: " + name);
      }
      input_names.push_back(name.c_str());
    }

    std::vector<std::string> output_name_storage;
    std::vector<const char*> output_names;
    for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
      auto allocated_name = session.GetOutputNameAllocated(i, allocator);
      output_name_storage.emplace_back(allocated_name.get());
    }
    for (const auto& name : output_name_storage) {
      output_names.push_back(name.c_str());
    }

    const std::uint64_t benchmark_rows = options.max_rows > 0
        ? std::min<std::uint64_t>(rows, static_cast<std::uint64_t>(options.max_rows))
        : rows;
    const std::array<std::int64_t, 2> shape{
        1, static_cast<std::int64_t>(sequence_length)};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    auto run_row = [&](std::uint64_t row) {
      std::vector<Ort::Value> tensors;
      tensors.reserve(input_names.size());
      for (const auto& name : input_name_storage) {
        auto& matrix = inputs.at(name);
        tensors.emplace_back(Ort::Value::CreateTensor<std::int64_t>(
            memory_info,
            matrix.row_ptr(row),
            static_cast<std::size_t>(sequence_length),
            shape.data(),
            shape.size()));
      }
      auto outputs = session.Run(
          Ort::RunOptions{nullptr},
          input_names.data(),
          tensors.data(),
          tensors.size(),
          output_names.data(),
          output_names.size());
      if (outputs.empty()) {
        throw std::runtime_error("ONNX model returned no outputs");
      }
    };

    const std::uint64_t warmup = std::min<std::uint64_t>(
        benchmark_rows, static_cast<std::uint64_t>(std::max(options.warmup_rows, 0)));
    for (std::uint64_t row = 0; row < warmup; ++row) {
      run_row(row);
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(benchmark_rows));
    const auto total_start = std::chrono::steady_clock::now();
    for (std::uint64_t row = 0; row < benchmark_rows; ++row) {
      const auto start = std::chrono::steady_clock::now();
      run_row(row);
      const auto end = std::chrono::steady_clock::now();
      latencies_ms.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
    }
    const auto total_end = std::chrono::steady_clock::now();
    const double total_seconds =
        std::chrono::duration<double>(total_end - total_start).count();
    std::sort(latencies_ms.begin(), latencies_ms.end());

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "{\n"
              << "  \"rows\": " << benchmark_rows << ",\n"
              << "  \"sequence_length\": " << sequence_length << ",\n"
              << "  \"threads\": " << options.intra_op_threads << ",\n"
              << "  \"p50_ms\": " << percentile(latencies_ms, 0.50) << ",\n"
              << "  \"p95_ms\": " << percentile(latencies_ms, 0.95) << ",\n"
              << "  \"p99_ms\": " << percentile(latencies_ms, 0.99) << ",\n"
              << "  \"qps_sequential\": "
              << static_cast<double>(benchmark_rows) / std::max(total_seconds, 1e-12) << "\n"
              << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
