#include "pybind_main_avx512.h"

#include "../../lib/formux.h"
#include "../../lib/simulator_avx512.h"
#include "../../lib/util_cpu.h"

namespace clfsim {
  template <typename For>
  using Simulator = SimulatorAVX512<For>;

  struct Factory {
    Factory (
        unsigned num_sim_threads,
        unsigned num_state_threads,
        unsigned num_dblocks
        ) : num_threads(num_sim_threads) {}

    using Simulator = clfsim::Simulator<For>;
    using StateSpace = Simulator::StateSpace;

    StateSpace CreateStateSpace() const {
      return StateSpace(num_threads);
    }

    StateSpace CreateStateSpace() const {
      return StateSpace(num_threads);
    }

    unsigned num_threads;
  };
}

#include "../pybind_main.cpp"
