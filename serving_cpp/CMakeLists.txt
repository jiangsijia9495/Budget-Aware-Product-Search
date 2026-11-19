cmake_minimum_required(VERSION 3.18)
project(budget_aware_search_serving LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(OpenMP QUIET)
find_package(faiss CONFIG QUIET)

add_executable(faiss_benchmark src/faiss_benchmark.cpp)
target_include_directories(faiss_benchmark PRIVATE include)

if(faiss_FOUND)
  target_link_libraries(faiss_benchmark PRIVATE faiss)
else()
  find_path(FAISS_INCLUDE_DIR faiss/Index.h HINTS ${FAISS_ROOT}/include)
  find_library(FAISS_LIBRARY NAMES faiss HINTS ${FAISS_ROOT}/lib ${FAISS_ROOT}/lib64)
  if(NOT FAISS_INCLUDE_DIR OR NOT FAISS_LIBRARY)
    message(FATAL_ERROR "FAISS not found. Set FAISS_ROOT or install a CMake-configured FAISS package.")
  endif()
  target_include_directories(faiss_benchmark PRIVATE ${FAISS_INCLUDE_DIR})
  target_link_libraries(faiss_benchmark PRIVATE ${FAISS_LIBRARY})
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(faiss_benchmark PRIVATE OpenMP::OpenMP_CXX)
endif()

option(BUILD_ONNX_BENCHMARK "Build the ONNX Runtime pair-scoring benchmark" ON)
if(BUILD_ONNX_BENCHMARK)
  find_path(
    ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
  find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64)
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    add_executable(onnx_benchmark src/onnx_benchmark.cpp)
    target_include_directories(onnx_benchmark PRIVATE include ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(onnx_benchmark PRIVATE ${ONNXRUNTIME_LIBRARY})
  else()
    message(WARNING "ONNX Runtime C++ not found; onnx_benchmark will not be built. Set ONNXRUNTIME_ROOT to enable it.")
  endif()
endif()
