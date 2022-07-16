#include <cublas_v2.h>
#include <custatevec.h>

#include "pybind_main_custatevec.h"
#include "../../lib/simulator_custatevec.h"

namespace  clfsim {

  using Simulator = SimulatorCustateVec<float>;

  struct Factory {
  using Simulator = clfsim::Simulator;
  using StateSpace = Simulator::StateSpace;

  Factory(
      unsigned num_sim_threads,
      unsigned num_state_threads,
      unsigned num_dblocks
      ) {
    ErrorCheck(cublasCreate(&cublas_handle));
    ErrorCheck(custatevecCreate(&custatevec_handle));
  }

  ~Factory() {
    ErrorCheck(cublasDestroy(cublas_handle));
    ErrorCheck(custatevecDestroy(custatevec_handle));
  }

  StateSpace CreateStateSpace() const {
    return StateSpace(cublas_handle, custatevec_handle);
  }

  Simulator CreateSimulator() const {
    return Simulator(custatevec_handle);
  }

  cublasHandle_t cublas_handle;
  custatevecHandle_t custatevec_handle;
};

inline void SetFlushToZerondDenormalsAreZeros() {}
inline void ClearFlushToZeroAndDenormalsAreZeros() {}

}

#include "../pybind_main.cpp"
