#ifndef FUSER_H_
#define FUSER_H_

#include <cstdint>
#include <vector>

#include "gate.h"
#include "matrix.h"

namespace clfsim {

/**
 * A collection of "fused" gates which can be multiplied together before being
 * applied to the state vector.
 */
template <typename Gate>
struct GateFused {
  /**
   * Kind of the first ("parent") gate.
   */
  typename Gate::GateKind kind;
  /**
   * The time index of the first ("parent") gate.
   */
  unsigned time;
  /**
   * A list of qubits these gates act upon. Control qubits for
   * explicitly-controlled gates are excluded from this list.
   */
  std::vector<unsigned> qubits;
  /**
   * Pointer to the first ("parent") gate.
   */
  const Gate* parent;
  /**
   * Ordered list of component gates.
   */
  std::vector<const Gate*> gates;
  /**
   * Fused gate matrix.
   */
  Matrix<typename Gate::fp_type> matrix;
};

/**
 * A base class for fuser classes with some common functions.
 */
template <typename IO, typename Gate>
class Fuser {
 protected:
  using RGate = typename std::remove_pointer<Gate>::type;

  static const RGate& GateToConstRef(const RGate& gate) {
    return gate;
  }

  static const RGate& GateToConstRef(const RGate* gate) {
    return *gate;
  }

  static std::vector<unsigned> MergeWithMeasurementTimes(
      typename std::vector<Gate>::const_iterator gfirst,
      typename std::vector<Gate>::const_iterator glast,
      const std::vector<unsigned>& times) {
    std::vector<unsigned> epochs;
    epochs.reserve(glast - gfirst + times.size());

    std::size_t last = 0;
    unsigned max_time = 0;

    for (auto gate_it = gfirst; gate_it < glast; ++gate_it) {
      const auto& gate = GateToConstRef(*gate_it);

      if (gate.time > max_time) {
        max_time = gate.time;
      }

      if (epochs.size() > 0 && gate.time < epochs.back()) {
        IO::errorf("gate crosses the time boundary.\n");
        epochs.resize(0);
        return epochs;
      }

      if (gate.kind == gate::kMeasurement) {
        if (epochs.size() == 0 || epochs.back() < gate.time) {
          if (!AddBoundary(gate.time, max_time, epochs)) {
            epochs.resize(0);
            return epochs;
          }
        }
      }

      while (last < times.size() && times[last] <= gate.time) {
        unsigned prev = times[last++];
        epochs.push_back(prev);
        if (!AddBoundary(prev, max_time, epochs)) {
          epochs.resize(0);
          return epochs;
        }
        while (last < times.size() && times[last] <= prev) ++last;
      }
    }

    if (epochs.size() == 0 || epochs.back() < max_time) {
      epochs.push_back(max_time);
    }

    return epochs;
  }

 private:
  static bool AddBoundary(unsigned time, unsigned max_time,
                          std::vector<unsigned>& boundaries) {
    if (max_time > time) {
      IO::errorf("gate crosses the time boundary.\n");
      return false;
    }

    boundaries.push_back(time);
    return true;
  }
};

/**
 * Multiplies component gate matrices of a fused gate.
 * @param gate Fused gate.
 */
template <typename FusedGate>
inline void CalculateFusedMatrix(FusedGate& gate) {
  MatrixIdentity(unsigned{1} << gate.qubits.size(), gate.matrix);

  for (auto pgate : gate.gates) {
    if (gate.qubits.size() == pgate->qubits.size()) {
      MatrixMultiply(gate.qubits.size(), pgate->matrix, gate.matrix);
    } else {
      unsigned mask = 0;

      for (auto q : pgate->qubits) {
        for (std::size_t i = 0; i < gate.qubits.size(); ++i) {
          if (q == gate.qubits[i]) {
            mask |= unsigned{1} << i;
            break;
          }
        }
      }

      MatrixMultiply(mask, pgate->qubits.size(), pgate->matrix,
                     gate.qubits.size(), gate.matrix);
    }
  }
}

/**
 * Multiplies component gate matrices for a range of fused gates.
 * @param gbeg, gend The iterator range [gbeg, gend) of fused gates.
 */
template <typename Iterator>
inline void CalculateFusedMatrices(Iterator gbeg, Iterator gend) {
  for (auto g = gbeg; g != gend; ++g) {
    if (g->kind != gate::kMeasurement) {
      CalculateFusedMatrix(*g);
    }
  }
}

/**
 * Multiplies component gate matrices for a vector of fused gates.
 * @param gates The vector of fused gates.
 */
template <typename FusedGate>
inline void CalculateFusedMatrices(std::vector<FusedGate>& gates) {
  CalculateFusedMatrices(gates.begin(), gates.end());
}

}  // namespace clfsim

#endif  // FUSER_H_
