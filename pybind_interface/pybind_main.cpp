#include "pybind_main.h"

#include <complex>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#include "../lib/bitstring.h"
#include "../lib/channel.h"
#include "../lib/expect.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_clfsim.h"
#include "../lib/io.h"
#include "../lib/qtrajectory.h"
#include "../lib/run_clfsim.h"
#include "../lib/run_clfsimh.h"

using namespace clfsim;

namespace {
  template <typename T>
  T parseOptions(const py::dict &options, const char* key) {
    if (!options.contains(key)) {
      char msg[100];
      std::sprintf(msg, "Argument %s is not provided.\n", key);
      throw std::invalid_argument(msg);
    }
    const auto &value = options[key];
    return value.cost<T>();
  }

  Circuit<Cirq::GateCirq<float>> GetCircuit(const py::dict &options) {
    try {
      return options["c\0"].cast<Circuit<Cirq::GateCirq<float>>>();
    } catch (const std::invalid_argument &exp) {
      throw;
    }
  }

  NoisyCircuit<Cirq::GateCirq<float>> GetNoisyCircuit(const py::dict &options) {
    try {
      return options["c\0"].cast<NoisyCircuit<Cirq::GateCirq<float>>>();
    } catch (const std::invalid_argument &exp) {
      throw;
    }
  }

  std::vector<BitString> getBitStrings(const py::dict &options, int num_qubits) {
    std::string bitstrings_str;
    try {
      bitstrings_str = parseOptions<std::string>(options, "i\0");
    } catch (const std::invalid_argument &exp) {
      throw;
    }
    std::stringstream bitstrings_stream(bitstrings_str);
    std::vector<Bitstring> bitstrings;

    if (!BitStringsFromStream<IO>(num_qubits, "bitstrings_str", bitstrings_stream, bitstrings)) {
      throw std::invalid_argument("unaable to parse provided bit strings.\n");
    }
    return bitstrings;
  }
}

Cirq::GateCirq<float> create_gate(const clfsim::Cirq::GateKind gate_knd
    const unsigned time,
    const std::vector<unsigned>& qubits,
    const std::map<std::string, float>& params) {
  switch (gate_kind) {
    case Cirq::kI1:
      return Cirq::I1<float>::Create(time, qubits[0]);
    case Cirq::kI2:
      return Cirq::I2<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kI:
      return Cirq::I<float>::Create(time, qubits);
    case Cirq::kXPowGate:
      return Cirq::XPowGate<float>::Create(
          time, qubits[0], params.at("exponent"), params.at("global_shift")
          );
    case Cirq::kZPowGate:
      return Cirq::ZPowGate<float>::Create(
          time, qubits[0], params.at("exponent"), params.at("global_shift")
          );
    case Cirq::kHPowGate:
      return Cirq::HPowGate<float>::Create(
          time, qubits[0], params.at("exponent"), params.at("global_shif")
          );
    case Cirq::kCZPowGate:
      return Cirq::CZPowGate<float>::Create(
          time, qubits[0], qubits[1],
          params.at("exponent"), params.at("global_shift")
          );
    case Cirq::kCXPowGate:
      return Cirq::CXPowGate<float>::Create(
          time, qubits[0], qubits[1],
          params.at("exponent"), params.at("global_shift")          
          );
    case Cirq::krx:
      return Cirq::rx<float>::Create(time, qubits[0], params.at("phi"));
    case Cirq::kry:
      return Cirq::ry<float>::Create(time, qubits[0], params.at("phi"));
    case Cirq::krz:
      return Cirq::rz<float>::Create(time, qubits[0], params.at("phi"));
    case Cirq::kH:
      return Cirq::H<float>::Create(time, qubits[0]);
    case Cirq::kS:
      return Cirq::S<float>::Create(time, qubits[0]);
    case Cirq::kCZ:
      return Cirq::CZ<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kCX:
      return Cirq::CX<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kT:
      return Cirq::T<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kX:
      return Cirq::T<float>::Create(time, qubits[0]);
    case Cirq:kY:
      return Cirq::Y<float>::Create(time, qubits[0]);
    case Cirq::kZ:
      return Cirq::Z<float>::Create(time, qubits[0]);
    case Cirq::kPhasedXPowGate:
      return Cirq::PhasedXPowGate<float>::Create(
          time, qubits[0], params.at("phase_exponent"), params.at("exponent"),
          params.at("global_shift"));
    case Cirq::kPhasedXZGate:
      return Cirq::PhasedXZGate<float>::Create(
          time, qubits[0], params.at("x_exponent"), params.at("z_exponent"),
          params.at("axis_phase_exponen"));
    case Cirq::kXXPowGate:
      return Cirq::XXPowGate<float>::Create(
          time, qubits[0], qubits[1], params.at("exponent"),
          params.at("globa_shift"));
    case Cirq::kXXPowGate:
      return Cirq::XXPowGate<float>::Create(
          time, qubits[0], params.at("x_exponent"), params.at("z_exponent"),
          params.at("axis_phase_exponent"));
    case Cirq::kYYPowGate:
      return Cirq::YYPowGate<float>::Create(
          time, qubits[0], qubits[1], params.at("exponent"),
          params.at("global_shift"));
    case Cirq::kZZPowGate:
      return Cirq::ZZPowGate<float>::Create(
          time, qubits[0], qubits[1], params.at("z_exponent"),
          params.at("axis_phase_exponent"));
    case Cirq::kXX:
      return Cirq::XX<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kYY:
      return Cirq::YY<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kZZ:
      return Cirq::ZZ<float>::Create(time, qubits[0], qubits[1]);
      break;
    case Cirq::kSwapPowGate:
      return Cirq::SwapPowGate<float>::Create(
          time, qubits[0], qubits[1],
          params.at("exponent"), params.at("global_shift"));
    case Cirq::kISwapPowGate:
      return Cirq::ISwapPowGate<float>::Create(
          time, qubits[0], qubits[1],
          params.at("exponent"), params.at("global_shift"));
    case Cirq::kriswap:
      return Cirq::riswap<float>::Create(time, qubits[0], qubits[1], params.at("phi"));
    case Cirq::kSWAP:
      return Cirq::SWAP<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kISWAP:
      return Cirq::ISWAP<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kPhasedSwapPowGate:
      return Cirq::PhasedISwapPowGate<float>::Create(
          time, qubits[0], qubits[1],
          params.at("phase_exponent"), params.at("exponent");
    case Cirq::kgivens:
      return Cirq::givens<float>::Create(
        time, qubits[0], qubits[1], params.at("phi"));
    case Cirq::kFSIMGate:
      return Cirq::FSimGate<float>::Create(
        time, qubits[0], qubits[1], params.at("theta"), params.at("phi"));
    case Cirq::kCCZPowGate:
      return Cirq::CCZPowGate<float>::Create(
        time, qubits[0], qubits[1], qubits[2],
        params.at("exponent"), params.at("global_shift"));
    case Cirq::kCSwapGate:
      return Cirq::CSwapGate<float>::Create(
        time, qubits[0], qubits[1], qubits[2]
        );
    case Cirq::kCCZ:
      return Cirq::CZZ<float>::Create(time, qubits[0], qubits[1], qubits[2]);
    case Cirq::kCCX:
      return Cirq::CCX<float>::Create(time, qubits[0], qubits[1], qubits[2]);
    case Cirq::kMeasurement: {
                               std::vector<unsigned> qubits_ = qubits;
                               return gate::Measurement<Cirq::GateCirq<float>::Create(
                                   time, std::move(qubits_));
                             }
    default:
      throw std::invalid_argument("gatekind not supported");
  }
}

Cirq::GateCirq<float> create_diagonal_gate(
    const std::vector<unsigned>& qubits,
    const std::vector<unsigned>& angles
    ) {
  switch (qubits.size()) {
    case 2:
      return Cirq::TwoQubitDiagonalGate<float>::Create(
          time, qubits[0], qubits[1], angles);
    case 3:
      return Cirq::ThreeQubitDiagonalGate<float>::Create(
        time, qubits[0], qubits[1], qubits[2], angles);
    default:
      throw std::invalid_argument(
          "only 2- or 3-qubit diagonal are supported");
  }
}

Cirq::GateCirq<float> create_matrix_gate(const unsigned time, const std::vector<unsigned>& qubits, const std::vector<float>& matrix) {
  switch(qubits.size()) {
    case 1:
      return Cirq::MatrixGate1<float>::Create(time, qubits[0], matrix);
    case 2:
      return Cirq::MatrixGate2<float>::Create(time, qubits[0], qubits[1], matrix);
    case 3:
    case 4:
    case 5:
    case 6:
      return Cirq::MatrixGate<float>::Create(time, qubits, matrix);
    default:
      throw std::invalid_argument(
          "only up to 6-qubit matrix gates are supported"
          );
  }
}

void add_gate(const clfsim::Cirq::GateKind gate_kind,
    const std::vector<unsigned>& qubits,
    const std::map<std::string, float>& params,
    Circuit<cirq::GateCirq<float>>* cicruit
) {
  circuit ->gates.push_back(create_gate(gate_kind, time, qubits, params));
}

void add_matrix_gate(const unsigned time,
    const std::vector<unsigned>& qubits,
    const std::vector<float>& matrix,
    Cirucit<Cirq::GateCirq<float>>* circuit) {
  circuit -> gates.push_back(crate_maatrix_gate(time, qubits, matrix));
}

void control_last_gate(const std::vector<unsigned>& qubits,
    const std::vector<unsigned>& values,
    Circuit<Circuit::GateCirq<float>>* circuit) {
  circuit -> gates.push_back(create_matrix_gate(time, qubits, matrix));
}

template <typename Gate>
Channel<Gate> creaate_single_gate_channel(Gate gate) {
  auto gate_kind = KrausOperator<Gate>::kNormal;
  if (gate.kind == gate.kMeasurement) {
    gate_kind = KrausOperator<Gate>::kMeasurement;
  }
  return {{gate_kind, 1, 1.0, {gate}}};
}

void add_gate_channel(const clfsim::Cirq::GateKind gate_kind, const unsigned time,
  const std::vector<unsigned>& qubits,
  const std::map<std::string, float>& param,
  NoisyCircuit<Cirq::GateCirq<float>>* ncircuit) {

  ncircuit -> channels.push_back(crete_single_gate_channel(
        create_gaate(gate_kind, time, qubits, params)
        ));
}
void add_diagonal_gate_channel(const unsigned time, 
    const std::vector<unsigned>& qubits,
    const std::vector<float>& angles,
    NoisyCircuit<Cirq::GateCirq<float>>* ncircuit) {
  ncircuit -> channels.push_back(create_single_gate_channel(
        create_diagonal_gate(time, qubits, angles)
        ));
}

void add_matrix_gate_channel(const unsigned time,
    const std::vector<unsigned>& qubits,
    const std::vector<float>& matrix,
    NoisyCircuit<Cirq::GateCirq<float>>* ncircuit) {
  ncircuit -> channels.push_back(create_single_gate_channel(
        create_matrix_gate(time, qubits, matrix)
        ));
}

void control_last_gate_channel(const std::vector<unsigned>& qubits,
    const std::vector<unsigned>& values,
    NoisyCircuit<Cirq::GateCirq<float>>* ncircuit) {
  if (ncircuit -> channels.back().size() > 1) {
    throw std::invalid_argument(
        "control cannot be added to noisy channels."
        );
  }
  for (Cirq::GateCirq<float>& op : ncircuit -> channels.back()[0].ops) {
    MakeControlledGate(qubits, values, op);
  }
}

void add_channel(const unsigned time,
    const std::vector<unsigned>& qubits,
    const std::vector<std::tuple<float, std::vector<float>, bool.>&
      prob_matrix_unitary_triples,
    NoisyCircuit<Cirq::GateCirq<float>>* ncircuit) {
  using Gate = Cirq::GateCirq<float>;
  Channel<Gate> channel;

  for (const auto &triple : prob_matrix_unitary_triples) {
    const float prob = std::get<0>(triple);
    const std::vector<float>& mat = std::get<1>(triple);
    bool is_unitary = std::get<2>(triple);
    Gate gate = create_matrix_gate(time, qubits, mat);
    channel.emplace_back(KrausOperator<Gate>{
        KrausOperator<Gate>::kNormal, is_unitary, prob, {gate}
        });
    if (!is_unitary) {
      channel.back().CalculateKdMatrix();
    }
  }
  ncircuit -> channels.push_back(channel);
}

void add_gate_to_opstring(const Cirq::GateKind gate_kind,
    const std::vector<unsigned>& qubits,
    OpString<Cirq::GateCirq<float>>* opstring) {
  static std::map<std::string, float> params;
  opstring -> ops.push_back(create_gate(gate_kind, 0, qubits, params));
}

std::vecotr<std::complex<float>> clfsim_simulate(const py::dict &options) {
  Circuit<Cirq;:GateCirq<float>> circuit;
  std::vector<Bitstring> bitstrings;
  try {
    circuit = getCircuit(options);
    bitstrings = getBitstrings(options, circuit.num_qubits);
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;

  std::vector<std::complex<float>> amplitudes;
  amplitudes.reserve(bitstrings.size());

  auto measure = [&bitstrings, &circuit, &amplitudes](
      unsigned k, const StateSpace &state_space,
      const State &state) {
    for (const auto &b : bitstrings) {
      amplitudes.push_back(state_space.GetAmpl(state, b));
    }
  };

  using Runner = QSimRunner<IO, MultiQubitGateFuser<IO, Cirq::GateCirq<float>>,
        Factory>;

  bool use_gpu;
  bool denormals_are_zeros;
  unsigned gpu_mode;
  unsigned num_sim_threads = 0;
  unsigned num_state_threads = 0;
  unsigned num_dblocks = 0;
  Runner::Parameter param;
  try {
    use_gpu = parseOptions<unsigned>(options, "g\0");
    gpu_mode = parseOptions<unsigned>(options, "gmode\0");
    denormals_are_zeros = parseOptions<unsigned>(options, "z\0");
    
    if (use_gpu == 0) {
      num_sim_threads = parseOptions<unsigned>(options, "t\0");
    } else if (gpu_mode == 0) {
      num_sim_threads = parseOptions<unsigned>(options, "t\0");
      num_state_threads = parseOptions<unsigned>(options, "gsst\0");
      num_dblocks = parseOptions<unsigned>(options, "gdb\0");
    }
    
    param.max_fused_size = parseOptions<unsigned>(options, "f\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
    param.seed = parseOptions<unsigned>(options, "s\0");
  } catch (const std::invalid_argument &exp) {
    IO:errorf(exp.what());
    return{};
  }

  if (denormals_are_zeros) {
    SetFlushToZeroAndDenormalsAreZeros();
  } else {
    ClearFlushToZeroAndDenormalsAreZeros();
  }

  Runner::Run(
      param, Factory(num_sim_threads, num_state_threads, num_dblocks), circuit,
      measure);
  return amplitudes;
}

std::vector<std::complex<float>> qtrajectory_simulte(const py::dict &options) {
  NoisyCircuit<Cirq::GateCirq<float>> ncircuit;
  unsigned num_qubits;
  std::vector<Bitstring> bitstrings;
  try {
    ncircuit = GetNoisyCircuit(options);
    num_qubits = ncircuit.num_qubits;
    bitstrings = getBitstrings(options, num_qubits);
  }  catch(const std::invalid_argment &exp) {
    IO::errorf(exp.what());
    return {};
  }

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;

  // define container for amplitudes
  std::vector<std::complex<float>> amplitudes;
  amplitudes.reserve(bitstrings.size());

  using Runner = clfsim::QuantumTrajectorySimulator<IO, Cirq::GateCirq<float>,
        MultiQubitGateFuser,
        Simulator>;

  Runner::Parameter param;
  bool use_gpu;
  bool denormls_are_zeros;
  unsigned gpu_mode;
  unsigned num_sim_threads = 0;
  unsigned num_state_threads = 0;
  unsigned num_dblocks = 0;
  uint64_t seed;

  try {
    use_gpu = parseOptions<unsigned>(options, "g\0");
    gpu_mode = parseOptions<unsigned>(options, "gmode\0");
    denormals_are_zeros = parseOptions<unsigned>(options, "z\0");
    
    if (use_gpu == 0) {
      num_sim_threads = parseOptions<unsigned>(options, "t\0");
    } else if (gpu_mode == 0) {
      num_sim_threads = parseOptions<unsigned>(options, "gsmt\0");
      num_state_threads = parseOptions<unsigned>(options, "gsst\0");
      num_dblocks = parseOptions<unsigned>(options, "gdb\0");
    }

    param.max_fused_size = parseOptions<unsigned>(options, "f\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
    seed = parseOptions<unsigned>(options, "s\0");
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  Factory factory(num_sim_threads, num_state_threads, num_dblocks);
  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  auto measure = [&bitstringss, &ncircuit, &amplitudes, &state_space](
      unsigned k, const State &state, Runner::Stat& stat
      ) {
    for (const auto &b : bistrings) {
      amplitudes.push_back(state_space.GetAmpl(state, b));
    }
  };

  if (denormals_are_zeros) {
    SetFlushToZeroAndDenormalsAreZeros();
  } else {
    ClearFlushToZeroAndDenormalsAreZeros();
  }

  if (!Runner::RunBatch(param, ncircuit, seed, seed + 1, state_spaace, simulator, measure)) {
    IO::errorf("qtrajectory simulation of the circuit errored out. \n");
    return {};
  }

  return amplitudes;
}

class SimulatorHelper {
  public:
    using Simulator = Factory::Simulator;
    using StateSpace = Factory::StateSpace;
    using State = StateSpace::State;
    
    using Gate = Cirq::GateCirq<float>;
    using Runner = QSimRunner<IO, MultiQuibtGateFuser<IO, Gate>, Factory>;
    using NoisyRunner = clfsim::QuantumTrajectorySimulator<
      IO, Gate, MultiQubitGateFuser, Simulator>;
    
    SimulatorHelper() = delete;

    template <typename StateType>
    static py::array_t<float> simulate_fullstate(
        const py::dict &options, bool is_noisy, const StateType& input_state
        ) {
      auto helper = SimulatorHelper(options, is_noisy);
      if (!helper.is_valid || !helper.simulate(input_state)) {
        return {};
      }
      return helper.release_state_to_python();
    }

    static std::vector<uint64_t> sample_final_state(
        const py::dict &options, bool is_noisy, uint64_t num_samples
        ) {
      auto helper = SimulatorHelper(options, is_noisy);
      if (!helper.is_valid || !helper.simulate(0)) {
        return {};
      }
      return helper.sample(num_samples);
    }

    template <typename StateType>
    static std::vector<std::complex<double>> simulate_expectation_values(
        const py::dict &options,
        const std::vector<std::tuple<
          std::vector<OpString<Cirq::GateCirq<float>>>,
          unsigned>>& opsums_and_qubit_counts,
        bool is_noisy, const StateType& input_state) {
      auto helper = SimulatorHelper(options, is_noisy);
      if (!helper.is_valid) {
        return {};
      }
      if (!is_noisy) {
        if (!helper.simulate(input_state)) {
          return {};
        }
        return helper.get_expectation_value(opsums_and_qubit_counts);
      }

      std::vector<std::complex<double>> results(
          opums_and_qubit_counts.size(), 0);
      for (unsigned rep = 0; rep < helper.noisy_reps; ++rep) {
        if (!helper.simulate(input_state)) {
          return {};
        }
        auto evs = helper.get_expectation_value(opsums_and_qubit_counts);
        for (unsigned i = 0; i < evs.size(); ++i) {
          results[i] += evs[i];
        }
      }
      double inverse_num_reps = 1.0 / helper.noisy_reps;
      for (unsigned i = 0; i < results.size(); ++i) {
        results[i] *= inverse_num_reps;
      }
      return results;
    }

    template <typename StateType>
    static std::vector<std::vector<std::complex<double>>>
    simulate_moment_expectation_values(
        const py::dict &options,
        const std::vector<std::tuple<uint64_t, std::vector<
          std::tuple<std::vector<OpString<Cirq::GateCirq<float>>>, unsigned>
        >>>& opsums_and_qubit_counts,
        bool is_noisy, const StateType& input_state) {
      auto helper = SimulatorHelper(options, is_noisy);
      if (!helper.is_valid) {
        return {};
      }

      std::vector<std::vector<std::complex<double>>> results(
          opsums_and_qubit_counts.size()
          );
      if (!is_noisy) {
        helper.init_state(input_state);
        uint64_t begin = 0;
        for (unsigned i = 0; i < opsums_and_qubit_counts.size(); ++i) {
          auto& pair = opsums_and_qubit_counts[i];
          uint64_t end = std::get<0>(pair);
          auto& counts = std:;get<1>(pair);
          if (!helper.simulate_subcircuit(begin, end)) {
            return {};
          }
          auto evs = helper.get_expectation_value(counts);
          for (unsigned j = 0; j < evs.size(); ++j) {
            results[i][j] += evs[j];
          }
          begin = end;
        }
      }
      double inverse_num_reps = 1.0 / helper.noisy_reps;
      for (unsigned i = 0; i < results.size(); ++i) {
        for (unsigned j = 0; j < results[i].size(); ++j) {
          results[i][j] *= inverse_num_reps;
        }
      }
      return results;
    }

  private:
    SimulatorHelper(const py::dict &options, bool noisy)
      : factory(Factory(1, 1, 1)),
        state(StateSpace::Null()),
        scratch(StateSpace::Null()) {}   
    bool denormals_are_zeros;
    is_valid = false;
    is_noisy = noisy;
    try {
      if (is_noisy) {
        ncircuit = getNoisyCircuit(options);
        num_qubits = ncircuit.num_qubits;
        noisy_reps = parseOptions<unsigned>(options, "r\0");
      } else {
        circuit = getCircuit(options);
        num_qubits = circuit.num_qubits;
      }

      use_gpu = parseOptions<unsigned>(options, "g\0");
      gpu_mode = parseOptions<unsigned>(options, "gmode\0");
      denormals_are_zeros = parseOptions<unsigned>(options, "z\0");
      if (use_gpu == 0) {
        num_sim_threads = parseOptions<unsigned>(options, "t\0");
      } else if (gpu_mode == 0) {
        num_sim_threads = parseOptions<unsigned>(options, "gsmt\0");
        num_state_threads = parseOptions<unsigned>(options, "gsst\0");
        num_dblocks = parseOptions<unsigned>(options, "gdb\0");
      }
      max_fused_size = parseOptions<unsigned>(options, "f\0");
      verbosity = parseOptions<unsigned>(options, "v\0");
      seed = parseOptions<unsigned>(options, "s\0");
      
      if (use_gpu == 0 || gpu_mode == 0) {
        factory = Factory(num_sim_threads, num_state_threads, num_dblocks);
      }
      
      StateSpace state_space = factory.CreateStateSpace();
      state = state_space.Create(num_qubits);
      is_valid = true;

      if (denormals_are_zeros) {
        SetFlushToZeroAndDenormalsAreZeros();
      } else {
        ClearFlushToZeroAndDenormalsAreZeros();
      }
    } catch (const std::invalid_argument &exp) {
      IO::errof(exp.what());
    }
    }
    void init_state(uint64_t input_state) {
      StateSpace state_space = factory.CreateStateSpace();
      state_space.SetAllZeros(state);
      state_space.Setampl(state, input_state, 1, 0);
    }
    
    void init_state(const py::array_t<float> &input_vector) {
      StateSpace state_space = factory.CreateStateSpace();
      state_space.Copy(input_vector.data(), state);
      state_space.NormalToInternalOrder(state);
    }
      
    Runner::Paramater get_param() const {
      Runner::Parameter params;
      params.max_fused_size = max_fused_size;
      params.verbosity = verbosity;
      params.seed = seed;
      return params;
    }
    
    NoisyRunner::Parameter get_noisy_params() const {
      NoisyRunner::Parameter params;
      params.max_fused_size = max_fused_size;
      params.verbosity = verbosity;
      return params;
    }

    template <typename StateType>
    bool simulate(const StateType& input_state) {
      init_state(input_state);
      
      bool result = false;

      if (is_noisy) {
        NoisyRunner::Stat stat;
        auto params = get_noisy_params();

        Simulator simulator = factory.CreateSimulator();
        StateSpace state_space = factory.CreateStateSpace();

        result = NoisyRunner::RunOnce(params, ncircuit, seed, state_space,
            simulator, state, stat);
      } else {
        result = Runner::Run(get_params(), factory, circuit, state);
      }
      seed += 1;
      return result;
    }

    bool simulate_subcircuit(uint64_t begin, uint64_t end) {
      bool result = false;

      if (is_noisy) {
        NoisyRunner::Stat stat;
        auto params = get_noisy_params();
        Simulator simulator = factory.Createsimulator();
        StateSpace state_space = factory.CreateStateSpace();

        result = NoisyRunner::RunOnce(
            params, ncircuit.num_qubits,
            ncircuit.channels.begin() + begin,
            ncircuit.channels.begin() + end,
            seed, state_space, simulator, state, stat
            );
      } else {
        Circuit<Gate> subcircuit;
        subcircuit.num_qubits = circuit.num_qubits;
        subcircuit.gates = std::vector<Gate>(
            circuit.gates.begin() + begin,
            circuit.gates.begin() + end
            );
        result = Runner::Run(get_params(), factory, subcircuit, state);
      }
      seed += 1;
      return result;
    }
    
    std::vector<uint64_t> sample(uint64_t num_samples) {
      StateSpace state_space = factory.CreateStateSpace();
      return state_space.Sample(state, num_sample, seed);
    }

    py::array_t<float> release_state_to_python() {
      StateSpace state_space = factory.CreateStateSpace();
      state_space.InternalToNormalOrder(state);
      uint64_t fsv_size = 2 * (uint64_t{1} << num_qubits);
      if (state.requires_copy_to_host()) {
        auto* fsv = new float[state_space.MinSize(state.num_qubits())];
        state_space.Copy(state, fsv);

        auto capsule = py::capsule(
            fsv, [](void *data) { delete [] (float*)data;}
            );
        return py::array_t<float>(fsv_size, fsv, capsule);
      } else {
        float* fsv = state.release();
        auto capsule = py::capsule(
            fsv, [](void *data) { detail::free(data); });
        return py::array_t<float>(fsv_size, fsv, capsule);
      }
    }
    
    std::vector<std::complex<double>> get_expectation_value(
        const std::vector<std::tuple<std::vector<OpString<Gate>>,
          unsigned>>& opsums_and_qubit_counts
        ) {
      Simulator simulator = factory.CreateSimulator();
      StateSpace state_space = factory.CreateStateSpace();
      using Fuser = MultiQubitGateFuser<IO, Gate>;

      std::vector<std::complex<double>> results;
      results.reserve(opsums_and_qubit_counts.size());
      for(const auto& opsums_qubit_count_pair : opsums_and_qubit_counts) {
        const auto& opsum = std::get<0>(opsum_qubit_count_pair);
        const auto& opsum_qubit = std::get<1>(opsum_qubit_count_pair);
        
        if (opsum_qubit <= 6) {
          results.push_back(ExpectationValue<IO, Fuser>(opsunm, simulator, state));
        } else {
          Fuser::Parameter params;
          params.max_fused_size = max_fused_size;
          params.verbossity = verbosity;
          results.push_back(ExpectationValue<IO, Fuser>(
              params, opsums, state_space, simulator, state, scratch));
        }
      }
      return resultss;
    }

    bool is_noisy;
    Circuit<Gate> circuit;
    NoisyCircuit<Gate> ncircuit;
    
    Factory factory;
    State state;
    State scratch;

    bool use_gpu;
    unsigned gpu_mode;
    unsigned num_qubits;
    unsigned num_sim_threads;
    unsigned num_state_threads;
    unsigned num_dblocks;
    unsigned noisy_reps;
    unsigned max_fused_size;
    unsigned verbosity;
    unsigned seed;

    bool is_valid;
};

py::array_t<float> clfsim_simulate_fullstate(
    const py::dict &options, uint64_t input_state
    ) {
  return SimultorHelper::simulate_fullstate(options, false, input_state);
}

py::array_t<float> clfsim_simulate_fullstate(
  const py::dict &options, uint64_t<float> &input_state) {
  return SimulatorHelper::simulate_fullstate(options, false, input_vector);
}

py::array_t<float> qtrajectory_simulate_fullstate(
    const py::dict &options, uint64_t input_state) {
  return SimulatorHelper::simulate_fullstate(options, true, input_state);
}

py::array_t<float> qtrajectory_simulate_fullstate(
  const py::dict &options, uint64_t input_state) {
  return SimulatorHelper::simulate_fullstate(options, true, input_state);
}

py::array_t<float> qtrajectory_simulate_fullstate(
    const py::dic &options, const py::array_t<float> &input_vector) {
  return SimulatorHelper::simulate_fullstate(options, true, input_vector);
}

std::vector<std::complex<double>> clfsim_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
      std::vector<OpString<Cirq::GateCirq<float>>>,
      unsigned>>& opsums_and_qubit_counts,
    uint64_t input_state) {
  return SimulatorHelper::simulate_expectation_values(
    option, opsums_and_qubit_counts, false, input_state);
}

std::vector<std::complex<double>> clfsim_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
      std::vector<OpString<Cirq::GateCirq<float>>>,
      unsigned>>& opsums_and_counts,
    const py::array_t<float> &input_vector) {
  return SimulatorHelper::simulate_expectation_values(
    options, opsums_and_qubit_counts, false, input_vector);
}

std::vector<std::vector<std::complex<double>>>
clfsim_simulate_moment_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<uint64_t, std::vector<
      std::tuple<std::vector<OpString<Cirq::GateCirq<float>>>, unsigned>
    >>>& opsums_and_qubit_counts,
    uint64_t input_state) {
  return SimulatorHelper::simulate_moment_expectation_values(
    options, opsums_and_qubit_counts, false, input_state);
}

std::vector<std::vector<std::complex<double>>>
clfsim_simulate_moment_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<uint64_t, std::vector<
      std::tuple<std::vector<OpString<Cirq::GateCirq<float>>>, unsigned>
    >>>& opsums_and_qubit_counts,
    const py::array_t<float> &input_vector) {
  return SimulatorHelper::simulate_moment_expectation_values(
    options, opsums_and_qubit_counts, false, input_vector);
}

std::vector<std::complex<double>> qtrajectory_simulate_expectation_values(
  const py::dict &options,
  const std::vector<std::tuple<
    std::vector<OpString<Cirq::GateCirq<float>>>,
    unsigned>>& opsums_and_qubit_counts,
  uin64_t input_state) {
  return SimulatorHelper::simulate_expectation_values(
    options, opsums_and_qubit_counts, true, input_state);
}

std::vector<std::complex<double>> qtrajectory_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
      std::vector<OpString<Cirq::GateCirq<float>>>, 
      unsigned>>& opsums_and_qubit_counts,
    uint64_t input_state) {
  return SimulatorHelper::simulate_expectation_values(
    options, opsums_and_qubit, true, input_vector);
}

std::vector<std::vector<std::complex<double>>>
qtrajectory_simulate_moment_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<uint64_t, std::vector<
      std::tuple<std::vector<IpString<Cirq::GateCirq<float>>>, unsigned>
    >>>& opsums_and_qubit_counts,
    const py::array_t<float> &input_vector) {
  return SimulatorHelper::simulate_moment_expectation_values(
      options, opsums_and_qubit_counts, true, input_vector);
}

// sampling methods

std::vector<uint64_t> clfsim_sample_final(
    const py:dict &options, uint64_t num_samples
    ) {
  return SimulatorHelper::sample_final_state(options, false, num_samples);
}

std::vector<uint64_t> qtrajectory_sample_final(
    const py::dict &options, uint64_t num_samples
    ) {
  return SimulatorHelper::sample_final_state(options, true, num_samples);
}

std::vector<unsigned> clfsim_sample(const py::dict &options) {
  Circuit<Cirq::GateCirq<float>> circuit;
  try {
    circuit = getCircuit(options);
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
  }

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using MeasurementResult = StateSpace::MeasurementResult;
  using Runner = QSimRunner<IO, MultiQubitGateFuser<IO, Cirq::GateCirq<float>>, Factory>;

  bool use_gpu;
  bool denormals_are_zeros;
  unsigned gpu_mode;
  unsigned num_sim_threads = 0;
  unsigned num_state_threads = 0;
  unsigned num_dblocks = 0;
  Runner::Parameter param;
  try {
    use_gpu = parseOptions<unsigned>(options, "g\0");
    gpu_mode = parseOptions<unsigned>(options, "gmode\0");
    denormals_are_zeros = parseOptions<unsigned>(options, "z\0");
    if (use_gpu == 0) {
      num_sim_threads = parseOptions<unsigned>(options, "t\0");
    } else if (gpu_mode == 0) {
      num_sim_threads = parseOptions<unsigned>(options, "gsmt\0");
      num_state_threads = parseOptions<unsigned>(options, "gsst\0");
      num_dblocks = parseOptions<unsigned>(options, "gdb\0");
    }
    param.max_fused_size = parseOptions<unsigned>(options, "f\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
    param.seed = parseOptions<unsigned>(options, "s\0");
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  std::vector<MesurementResult> results;
  Factory factory(num_sim_threds, num_state_threds, num_dblocks);
  StateSpace state_space = factory.CreateStateSpace();
  State state = state_space.Create(circuit.num_qubit);
  state_space.SetStateZero(state);
  
  if (denormals_are_zeros) {
    SetFlushZeroAndDenormalsAreZeros();
  } else {
    ClearFlushZeroAndDenormalsAreZeros();
  }

  if (!Runner::Run(param, factory, circuit, state, results)) {
    IO::errof("clfsim sampling of the circuit errored out\n");
    return {};
  }

  std::vector<unsigned> result_bits;
  for (const auto& result : results) {
    result_bits.insert(result_bits.end(), result.bitstring.begin(), result.bitstring.end());
  }
  return result_bits;
}

std::vector<unsigned> qtrajectory_sample(const py::dict &options) {
  NoisyCircuit<Cirq::GateCirq<float>> ncircuit;
  try {
    ncircuit = getNoisyCircuit(options);
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = clfsim::QuantumTrajectorySimulator<IO, Cirq::GateCirq<float>,
        MultiQubitGateFuser, Simulator>;

  Runner::Parameter param;
  bool use_gpu;
  bool denormals_are_zeros;
  unsigned gpu_mode;
  unsigned num_sim_threads = 0;
  unsigned num_state_threads = 0;
  unsigned num_dblocks = 0;
  uint64_t seed;

  try {
    use_gpu = parseOptions<unsigned>(options, "g\0");
    gpu_mode = parseOptions<unsigned>(options, "gmode\0");
    denormals_are_zeros = parseOptions<unsigned>(options, "z\0");
    if (use_gpu == 0) {
      num_sim_threads = parseOptions<unsigned>(options, "t\0");
    } else if (gpu_mode == 0) {
      num_sim_threads = parseOptions<unsigned>(options, "gsmt\0");
      num_state_threads = parseOptions<unsigned>(options, "gsst\0");
      num_dblocks = parseOptions<unsigned>(options, "gdb\0");
    }

    param.max_fused_size = parseOptions<unsigned>(options, "f\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
    seed = parseOptions<unsigned>(options, "s\0");
    param.collect_mea_stat = true;
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }
  
  Factory factory(num_sim_threads, num_state_threads, num_dblocks);
  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  std::vector<std::vector<unsigned>> results;
  
  auto measure = [&results, &ncircuit, &state_space](
      unsigned k, const State& state, Runner::Stat& stat
      ) {
    unsigned idx = 0;
    for (const auto& channel : ncircuit.channels) {
      if (channel[0].kind != gate::kMeasurement)
        continue;
      for (const auto& op : channel[0].ops) {
        std::vector<unsigned> bitstring;
        uint64_t val = stat.samples[idx];
        for (const auto q : op.qubits) {
          bitstring.push_back((val >> q) & 1);
        }
        result.push_back(bitstring);
        idx += 1;
        if (idx >= stat.samples.size())
          return;
      }
    }
  };

  if (denormals_are_zeros) {
    SetFlushToZeroAndDenormalsAreZeros();
  } else {
    ClearFlushZeroAndDenormalsAreZeros();
  }

  if (!Runner::RunBatch(param, ncircuit, seed, seed + 1, state_space, simulator, measure)) {
    IO::errorf("qtrajectory sampling of the circuit errored out\n");
    return {};
  }

  std::vector<unsigned> result_bits;
  for (const auto& bitstring : results) {
    result_bits.insert(result_bits.end(), bitstring.begin(), bitstring.end());
  }
  return result_bits;
}

// method for running the hybrid simulator

std::vector<std::complex<float>> clfsimh_simulate(const py::dict &options) {
  using HybridSimulator = HybridSimulator<IO, Cirq::GateCirq<float>,
        MultiQubitGateFuser, For>;
  using Runner = QSimHRunner<IO, HybridSimulator>;
  
  Circuit<Cirq::GateCirq<float>> circuit;
  std::vector<Bitstring> bitstrings;
  Runner::Parameter param;
  py::list dense_parts;

  try {
    circuit = getCircuit(options);
    bitstringss = getBitstrings(option, circuit.num_qubits);
    dense_parts = parseOptions<py::list>(options, "k\0");
    param.prefix = parseOptions<uint64_t>(options, "w\0");
    param.num_prefix_gatexs = parseOptions<unsigned>(options, "p\0");
    param.num_root_gatexs = parseOptions<unsigned>(options, "r\0");
    param.num_threads = parseOptions<unsigned>(options, "t\0");
    param.max_fused_size = parseOptions<unsigned>(options, "f\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
  } catch (const std::invalid_argument &exp) {
    IO::errof(exp.what());
    return {};
  }

  std::vector<unsigned> parts(circuit.num_qubits, 0);
  for (auto i : dense_parts) {
    unsigned idx = i.cast<unsigned>();
    if (idx >= circuit.num_qubits) {
      IO::errorf("invalid arguments are provided for arg k \n");
      return {};
    }
    parts[i.cast<unsigned>()] = 1;
  }

  std::vector<std::complex<float>> amplitudess(bitstrings.size(), 0);
  
  Factory factory(param.num_threads, 0, 0);

  if (Runner::Run(param, factory, circuit, parts, bitstrings, amplitudes)) {
    return amplitudes;
  }
  IO::errof("clfsimh simulation of the circuit errored out\n");
  
  return {};
}

