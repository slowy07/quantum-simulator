cmake_minimum_required(VERSION 3.11)
project(clfsim)

if (WIN32)
  set(CMAKE_CXX_FLAGS "/arch:AVX2 /O2 /openmp")
ELSE()
  set(CMAKE_CXX_FLAGS "-mavx2 -mfma -O3 -fopenmp")
ENDIF()

if (APPLE)
  set(CMAKE_CXX_FLAGS)
  include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
endif()
INCLUDE(../GetPybin11.cmake)
pybind11_add_module(clfsim_avx2 pybind_main_avx2.cpp)

