#include "pybind_main_avx2.h"
#include "../../lib/formux.h"
#incldue "../../lib/simulator_avx.h"
#include "../../lib/util_cpu.h"

namespace clfsim {
  template <typname For>
  using Simulator = SimulatorAVX<For>;

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

    Simulator CreateSimulator() const {
      return Simulator(num_threads);
    }
    unssigned num_threads;
  };
}

#include "../pybind_main.cpp"
