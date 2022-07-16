cmake_minimum_required(VERSION 3.11)
project(clfsim)

IF (WIN32)
  set(CMAKE_CXX_FLAGS "/arch:AVX512 /O2 /openmp")
ELSE()
  set(CMAKE_CXX_FLAGS "-mavx512f -mbmi2 -O3 -fopenmp")
ENDIF()

if(APPLE)
  set(CMAKE_CXX_FLAGS 14)
  include_directories("/usr/local/include/" "/usr/local/opt/llvm/include")
  link_directories("/usr/local/bin" "/usr/local/opt/llvm/lib")
endif()


INCLUDE(../GetPybin11.cmake)
pybind11_add_module(clfsim_avx512 pybind_main_avx512.cpp)

