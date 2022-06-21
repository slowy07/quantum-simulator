#ifndef CIRCUIT_H_
#define CIRCUIT_H_

#include <vector>

namespace clfsim {
  // A collection of gates.
  template <typename Gate>
  struct Circuit {
    unsigned num_qubits;
    std::vector<Gate> gates;
  }
}

#endif
