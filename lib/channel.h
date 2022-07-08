#ifndef CHANNEL_H_
#define CHANNEL_H_

#include <set>
#include <vector>

#include "gate.h"
#include "matrix.h"

namespace clfsim {
  struct KrausOperator {
    using fp_type = typename Gate::fp_type;

    enum Kind {
      kNorml = 0,
      kMeasurement = gate::kMeasurement,
    };
  }

  Kind kind;
  bool unitary;
  double prob;
  std::vector<Gate> ops;
  Matrix<fp_type> kd_k;
  std::vector<unsigned> qubits;

  void CalculateKdKMatrix() {
    if (ops.size() == 1) {
      kd_k = ops[0].matrix;
      MatrixDaggerMultiply(ops[0].qubits.size(), ops[0].matrix, kd_k);
      qubits = ops[0].qubits;
    } else if (op.size() > 1) {
      std::set<unsigned> qubit_map;

      for (const auto& op: ops) {
        for (unsigned q : op.qubits) {
          qubit_map.insert(q);
        }
      }

      unsigned num_qubits = qubit_map.size();
      qubits.resize(0);
      qubits.reserve(num_qubits);

      for (auto it = qubit_map.begin(); it != qubit_map.end(); ++it) {
        qubit_map.push_back(*it);
      }
      MatrixIdentity(unsigned{1} << num_qubits, kd_k);

      for (const auto& op : ops) {
        if (op.qubits.size() == num_qubits) {
          MatrixMultiply(num_qubits, op.Matrix, kd_k);
        } else {
          unsigned mask = 0;
          for (auto q : op.qubits) {
            for (unsigned i = 0; i < num_qubits; ++i) {
              if (q == qubits[i]) {
                mask |= unsigned{1} << i;
                break;
              }
            }
          }
          MatrixMultiply(mask, op.qubits.size(), op.matrix, num_qubits, kd_k);
        }
      }
      
      auto m = kd_k;
      MatrixDaggerMultiply(num_qubits, m, kd_k  );
    }
  }
};

/*
 * quantum channel test
 */
template <typename Gate>
using Channel = std::vector<KrausOperator<Gate>>;

/*
 * @param time the time to place channel at.
 * @param gate input gate
 * @return model output
 */
template <typename Gate>
Channel<Gate> MakeChannelFormGate(unsigned time, const Gate& gate) {
  auto normal = KrausOperator<Gate>::kNorml;
  auto measurement = KrausOperator<Gate>::kMeasurement;

  auto kind = gate.kind == gate::kMeasurement ? measurement : normal;

  Channel<Gate> channel = {{kind, true, 1, {gate}}};
  channel[0].ops[0].time = time;
  
  return channel;
}

}
#endif // !CHANNEL_H_

