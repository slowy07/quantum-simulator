cmake_minimum_required(VERSION 3.11)

IF (WIN32)
  set(CMAKE_CXX_FLAGS "/O2 /openmp")
ELSE()
  set(CMAKE_CXX_FLAGS "-O3 -fopenmp")
ENDIF()

if(APPLE)
  set(CMAKE_CXX_STANDARD 14)
  include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
  link_directories("/usr/local/lib/" "/usr/local/opt/llvm/lib")
endif()

INCLUDE(../GetPybin11.cmake)
find_package(PythonLibs 3.6 REQUIRED)

