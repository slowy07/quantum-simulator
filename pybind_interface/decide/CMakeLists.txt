cmake_minimum_required(VERSION 3.11)

execute_process(COMMAND which nvcc OUTPUT_VARIABLE has_nvcc)
if(has_nvcc STREQUAL "")
  project(clfsim)
else()
  project(clfsim LANGUAGES CXX CUDA)
endif()

if(APPLE)
  set(CMAKE_CXX_STANDARD 14)
  include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
  link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()

INCLUDE(../GetPybind11.cmake)

if(has_nvcc STREQUAL "")
  pybind11_add_module(clfsim_decide decide.cpp)
else()
  find_package(PythonLibs 3.6 REQUIRED)
  find_package(CUDA REQUIRED)

  include_directories(${PYTHON_INCLUDE_DIRS} ${pybind11_SOURCE_DIR}/include)

  cuda_add_library(clfsim_decide MODULE decide.cpp)

  if(DEFINED ENV{CUQUANTUM_DIR})
      target_compile_options(clfsim_decide PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-D__CUSTATEVEC__>
      )
  endif()

  set_target_properties(clfsim_decide PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
  )
  set_source_files_properties(decide.cpp PROPERTIES LANGUAGE CUDA)
endif()
