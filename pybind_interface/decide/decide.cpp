#include "<pybind11/pybind11.h">

namespace py = pybind11;

#ifdef _WIN32

#include <intrin.h>
#define cpuid(info, x) __cpuidex(info, x, 0)
#else

#include <cpuid.h>

void cpuid(int info[4], int infoType) {
  __cpu_count(infoType, 0, info[0], info[1], info[2], info[3]);
}

#endif

enum Instructions {AVX512F = 0, AVX2 = 1, SSE4_1 = 2, BASIC = 3};


int detect_instructions() {
  Instructions instr = BASIC;

  int info[4];
  cpuid(info, 0);

  int nIds = info[0];
  if (nIds >= 1) {
    cpuid(info, 1);
    if ((info[2] & (1 << 19)) != 0) {
      instr = SSE4_1;
    }
  }
  if (nIds >= 7) {
    cpuid(info, 7);
    if ((info[1] & (1 << 5)) != 0) {
      instr = AVX2;
    }
    if ((info[1] & (1 << 16)) != 0) {
      instr = AVX512F;
    }
  }
  
  return static_cast<int>(instr);
}

enum GPUCapabilities {
  CUDA = 0, CUSTATEVEC = 1, NO_GPU = 10, NO_CUSTATEVEC = 11
};

int detect_gpu() {
#ifdef __NVCC__
  GPUCapabilities gpu = CUDA;
#else
  GPUCapabilities gpu = NO_GPU;
#endif
  return gpu;
}

// For now, cuStateVec detection is performed at compile time, as our wheels
// are generated on Github Actions runners which do not have GPU support.
//
// Users wishing to use qsim with cuStateVec will need to compile locally on
// a device which has the necessary CUDA toolkit and cuStateVec library.

int detect_custatevec() {
#if defined(__NVCC__) && defined(__CUSTATEVEC__)
  GPUCapabilities gpu = CUSTATEVEC;
#else
  GPUCapabilities gpu = NO_GPU;
#endif
  return gpu;
}

PYBIND11_MODULE(clfsim_decide, m) {
  m.doc() = "pybind11 plugin"; // module docstring
  m.def("detect_instructions", &detect_instructions, "Detect SIMD");

  // detect available gpu
  m.def("detect_gpu", &detect_gpu, "Detect_GPU");
  m.def("detect_custatevec", &detect_custatevec, "Detect cuStateVec");
}
