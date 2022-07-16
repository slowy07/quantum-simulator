#include "pybind_main_cuda.h"
#include "../../lib/simulator_cuda.h"

namespace clfsim {
  using Simulator = SimulatorCUDA<float>;

  using Factory {
    using Simulator = clfsim::Simulator;
    using StateSpace = Simulator::StateSpace;

    Factory(
        unsigned num_sim_threads,
        unsigned num_state_threads,
        unsigned num_dblocks
        ) : ss_params{num_state_threads, num_dblocks},
      sim_params{num_sim_threads} {}

    StateSpace CreateStateSpace() const {
      return StateSpace(ss_params);
    }

    Simulator CreateSimulator() const {
      return Simulator(sim_params);
    }

    StateSpace::Parameter ss_params;
    Simulator::Parameter sim_params;
  };

  inline void SetFlushToZeroAndDenormalsAreZeros() {}
  inline void ClearFlushToZeroAndDenormalsAreZeros() {}
}

#include "../pybind_main.cpp"
