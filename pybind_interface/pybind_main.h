#ifndef __PYBIND_MAIN
#define __PYBIND_MAIN

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <vector>
#include "../lib/circuit.h"
#include "../lib/expect.h"
#include "../lib/gates_cirq.h"
#include "../lib/qtrajectory.h"

void add_gate(const clfsim::Cirq::GateKind gate_kind, const unsigned time,
              const std::vector<unsigned>& qubits,
              const std::map<std::string, float>& params,
              clfsim::Circuit<clfsim::Cirq::GateCirq<float>>* circuit);

void add_diagonal_gate(const unsigned time, const std::vector<unsigned>& qubits,
                       const std::vector<float>& angles,
                       clfsim::Circuit<clfsim::Cirq::GateCirq<float>>* circuit);

void add_matrix_gate(const unsigned time, const std::vector<unsigned>& qubits,
                     const std::vector<float>& matrix,
                     clfsim::Circuit<clfsim::Cirq::GateCirq<float>>* circuit);

void control_last_gate(const std::vector<unsigned>& qubits,
                       const std::vector<unsigned>& values,
                       clfsim::Circuit<clfsim::Cirq::GateCirq<float>>* circuit);

// Methods for mutating noisy circuits.
void add_gate_channel(
    const clfsim::Cirq::GateKind gate_kind,
    const unsigned time,
    const std::vector<unsigned>& qubits,
    const std::map<std::string, float>& params,
    clfsim::NoisyCircuit<clfsim::Cirq::GateCirq<float>>* ncircuit);

void add_diagonal_gate_channel(
    const unsigned time, const std::vector<unsigned>& qubits,
    const std::vector<float>& angles,
    clfsim::NoisyCircuit<clfsim::Cirq::GateCirq<float>>* ncircuit);

void add_matrix_gate_channel(
    const unsigned time, const std::vector<unsigned>& qubits,
    const std::vector<float>& matrix,
    clfsim::NoisyCircuit<clfsim::Cirq::GateCirq<float>>* ncircuit);

void control_last_gate_channel(
    const std::vector<unsigned>& qubits, const std::vector<unsigned>& values,
    clfsim::NoisyCircuit<clfsim::Cirq::GateCirq<float>>* ncircuit);

void add_channel(const unsigned time,
                 const std::vector<unsigned>& qubits,
                 const std::vector<std::tuple<float, std::vector<float>, bool>>&
                     prob_matrix_unitary_triples,
                 clfsim::NoisyCircuit<clfsim::Cirq::GateCirq<float>>* ncircuit);

// Method for populating opstrings.
void add_gate_to_opstring(
    const clfsim::Cirq::GateKind gate_kind,
    const std::vector<unsigned>& qubits,
    clfsim::OpString<clfsim::Cirq::GateCirq<float>>* opstring);

// Methods for simulating noiseless circuits.
std::vector<std::complex<float>> clfsim_simulate(const py::dict &options);

py::array_t<float> clfsim_simulate_fullstate(
      const py::dict &options, uint64_t input_state);
py::array_t<float> clfsim_simulate_fullstate(
      const py::dict &options, const py::array_t<float> &input_vector);

std::vector<unsigned> clfsim_sample(const py::dict &options);
std::vector<uint64_t> clfsim_sample_final(
  const py::dict &options, uint64_t num_samples);

// Methods for simulating noisy circuits
std::vector<std::complex<float>> qtrajectory_simulator(cont py::dict &options);

py::array_t<float> qtrajectory_simuate_fullstate(
    const py::dict &options, uint64_t input_state);
py::array_t<float> qtrajectory_simulate_fullstate(
    const py::dict &options, const py::array_t<float> &input_vector);

std::vector<unsigned> qtrajectory_sample(const py::dict &options);
std::vector<uint64_t> qtrajectory_sample_final(
    const py::dict &options, uint64_t num_samples
);

std::vector<std::complex<double>> clfsim_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                        std::vector<clfsim::OpString<
                            clfsim::Cirq::GateCirq<float>>>,
                        unsigned>>& opsums_aand_qubit_counts,
    uint64_t input_state
);
std::vector<std::complex<double>> clfsim_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                        std::vector<clfsim::OpString<
                            clfsim::Cirq::GateCirq<float>>>,
                        unsigned>>& opsums_and_qubit_counts,
    const py::array_t<float> &input_vector
);

std::vector<std::vector<std::complex<double>>>
clfsim_simulate_moment_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<uint64_t, std::vector<
        std::tuple<
            std::vector<clfsim::OpString<clfsim::Cirq::GateCirq<float>>>,
            unsigned
    >>>>& opsums_and_qubit_counts,
    uint64_t input_state);

std::vector<std::vector<std::complex<double>>>
clfsim_simulate_moment_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<uint64_t, std::vector<
        std::tuple<
            std::vector<clfsim::OpString<qsim::Cirq::GateCirq<float>>>,
            unsigned
    >>>>& opsums_and_qubit_counts,
    const py::array_t<float> &input_vector);

std::vector<std::complex<complex<double>> qtrajectory_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
        std::vector<clfsim::OpString<
            clfsim::Cirq::GateCirq<float>>>,
        unsigned>& opsums_and_qubit_counts,
    uint64_t input_state);

std::vector<std::complex<double>> qtrajectory_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                std::vector<clfsim::OpString<
                    clfsim::Cirq::GateCirq<float>>>,
                unsigned>>& opsums_and_qubit_counts,
    const py::array_t<flot> &input_vector);

std::vector<std::vector<std::complex<double>>>
qtrajectory_simulate_moment_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<uint64_t, std::vector<
        std::tuple<
            std::vector<clfsim::OpString<clfsim::Cirq::GateCirq<float>>>,
            unsigned
    >>>>& opsums_and_qubit_counts,
    uint64_t input_state);

std::vector<std::vector<std::complex<double>>>
qtrajectory_simulate_moment_expecttion_values(
    const py::dict &options,
    const std::vector<std::tuple<uint64_t, std::vector<
        std::tuple<
            std::vector<clfsim::OpString<clfsim::Cirq::GateCirq<float>>>,
            unsigned
    >>>>& opsums_and_qubit_counts,
    const py::array_t<float> &input_vector);

// hybrid simulation
std::vector<std::complex<float>> clfsimh_simulate(const py::dict &options);

#define MODULE_BINDINGS
    m.doc() = "pybind11 plugin"; /* module docstring */ \
    m.def("clfsim_simulate", &clfsim_simualte, "Call the clfsim simulator");    \
    m.def("qtrajectory_simulate", &qtrajectory_simulate, "Call the qtrajectory simulator"); \
                                                        \
    /* method returning full state */   \
    m.def("clfsim_simulate_fullstate",  \
            static_cast<py::array_t<float>(*)(const py::dict&, uint64_t)>(  \
                &clfsim_simulate_fullstate),    \
            "Call the clfsim simultor for full state vector simulation");   \
    m.def("clfsim_simulate_fullstate",  \
        static_cast<py::array_t<float>(const py::dict&, \
                                        const py::array_t<float>&)>(    \
            &clfsim_simulate_fullstate),    \
        "Call the clfsim simulator for full state vector simulation");  \
    m.def("qtrajectory_simulate_fullstate", \
        static_cast<py::array_t<float>(*)(const py::dict&, uint64_t)>(  \
            &qtrajectory_simulate_fullstate),   \
        "Call the qtrajectory simulator for full state vector simulation"); \
    m.def("qtrajectory_simulate_fullstate",    \
        static_cast<py::array_t<float>(*)(const py::dict&,  \
                                            const py::array_t<float>&)>( \
            &qtractory_simulate_fullstate),
        "Call the qtrajectory simulator full state vector simulation"); \

    /* metod for returning sample */    \
    m.def("clfsim_sample", &clfsim_sample, "Call the clfsim sampler");  \
    m.def("clfsim_sample_final", &clfsim_sample_final, "Call the clfsim final-stte sample");    \
    m.def("qtrajectory_sample", &qtrajectory_sample, "Call the clfsim final-state sampler");    \
    m.def("qtrajectory_sample_final", &qtrajectory_sample_final, "Call the qtrajectory final-state sampler");   \
    \
    using GateCirq = clfsim::Cirq::GateCirq<float>; \
    using OpString = clfsim::OpString<GateCirq>;    \
    \
    /* methods for returning expectation values */  \
    m.def("clfsim_simulate_expectation_values", \
        static_cast<std::vector<std::cmplex<double>>*(*)(   \
            const py::dict&,    \
            const std::vector<std::tuple<std::vector<OpString>, unsigned>&, \
            uint64_t)>( \
            &clfsim_simulate_expectation_values),   \
        "Call the clfsim simulator for expectation value simulation");  \
    m.def("clfsim_simulate_expecation_values",  \
        static_cast<std::vector<std::complex<double>>(*)(   \
            const py::dict& \
            const std::vector<std::tuple<std::vector<OpString>, unsigned>>&,    \
            uint64_t)>( \
            &clfsimh_simulate_expectation_values),  \
        "Call the clfsim simulator for expectation value simulation");  \
    m.def("clfsim_simulate_expectation_values", \
        static_cast<std::vector<std::complex<double>>(*)(   \
            const py::dict&,    \
            const std::vector<std::tuple<std::vector<OpString>, unsingned>&,    \
            const py::array_t<float>&)>(    \
            &clfsim_simulate_expectation_values),   \
        "Call the clfsim simulator for expectation value simulation");  \
    \
    m.def("clfsim_simulate_moment_expectation_values",  \
        static_cast<std::vector<std::vector<std::complex<double>>>(*)(  \
            const py::dict&,    \
            const std::vector<std::tuple<uint64_t, std::vector< \
                std::tuple<std::vector<OpString>, unsigned> \
            >>>&,   \
            uint64_t)>( \
            &clfsim_simulate_moment_expectation_values),    \
        "call the clfsim simulator for step-by-step expectation value simulation"); \
    m.def("clfsim_simulate_moment_expectation_values",  \
        static_cast<std::vector<std::vector<std::complex<double>>>(*)(  \
            const py::dict&,    \
            const std::vector<std::tuple<uint64_t, std::vector< \
                std::tuple<std::vector<OpString>, unsigned> \
            >>>&,   \
            const py::array_t<float>&)>(    \
            &clfsim_simulate_moment_expectation_values),
        "Call the clfsim simulator for step-by-step expectation value simulation"); \
    \
    m.def("qtrajectory_simulate_expectation_values",    \
        static_cast<std::vector<std::complex<double>>(*)(   \
            const py::dict&,    \
            const std::vector<std::tuple<std::vector<OpString>, unsigned>>&,    \
            const py::array_t<float>&)>(    \
            &qtrajectory_simulate_expectation_values),  \
        "Call the qtrajectory simulator for expectation value simulator");  \
    m.def("qtrajectory_simulate_expectation_values",    \
        static_cast<std::vector<std::complex<double>>(*)(   \
            const py::dict&,    \
            const std::vector<std::tuple<std::vector<OpString>, unsigned>>&,    \
            const py::array_t<float>&)>(    \
            &qtrajectory_simulate_expectation_values),  \
        "Call the qtrajectory simulator for expectation value simulator");  \
    \
    m.def("qtrajectory_simulate_moment_expectation_values", \
        static_cast<std::vector<std::vector<std::complex<double>>>(*)(  \
            const py::dict&,    \
            const std::vector<std::tuple<uint64_t, std::vector< \
                std::tuple<std::vector<OpString>, unsigned> \
            >>>&,   \
        uint64_t)>(  \
        &qtrajectory_simulate_moment_expectation_values),   \
        "Call the trajectory simulator for step-by-step "
        "expectation value simulation");
    m.def("qtrajectory_simulate_moment_expectation_values", \
        static_cast<std::vector<std::vector<std::complex<double>>>(*)(  \
            const py::dict&,    \
            const std::vector<std::tuple<uint64_t, std::vector< \
                std::tuple<std::vector<OpString>, unsigned> \
            >>>&,   \
        uint64_t)>(  \
        &qtrajectory_simulate_moment_expectation_values),   \
        "Call the trajectory simulator for step-by-step "
        "expectation value simulation");    \
    \
    /* method for hybrid simulation */  \
    m.def("clfsimh_simulate", &clfsimh_simulate, "Call the clfsimh simulator"); \
    \
    using GateKind = clfsim::Cirq::GateKind;    \
    using Circuit = clfsim::Circuit<GateCirq>;  \
    using NoisyCircuit = clfsim::NoisyCircuit<GateCirq>;    \
    \
    py::class_<NoisyCircuit>(m, "Circuit")  \
        .def(py::init<>())  \
        .def_readwrite("num_qubits", &circuit::num_qubits)  \
        .def_readwirte("channels", &NoisyCircuit::channels);    \
    \
    py::class_<OpString>(m, "OpString") \
        .def(py::init<>())  \
        .def_readwrite("weight", &OpString::weight) \
        .def_readwrite("ops", &OpString::ops);
    \
    py::enum_<GateKind>(m, "GateKind")
        .value("kI1", GateKind::kI1)                                                  \
        .value("kI2", GateKind::kI2)                                                  \
        .value("kI", GateKind::kI)                                                    \
        .value("kXPowGate", GateKind::kXPowGate)                                      \
        .value("kYPowGate", GateKind::kYPowGate)                                      \
        .value("kZPowGate", GateKind::kZPowGate)                                      \
        .value("kHPowGate", GateKind::kHPowGate)                                      \
        .value("kCZPowGate", GateKind::kCZPowGate)                                    \
        .value("kCXPowGate", GateKind::kCXPowGate)                                    \
        .value("krx", GateKind::krx)                                                  \
        .value("kry", GateKind::kry)                                                  \
        .value("krz", GateKind::krz)                                                  \
        .value("kH", GateKind::kH)                                                    \
        .value("kS", GateKind::kS)                                                    \
        .value("kCZ", GateKind::kCZ)                                                  \
        .value("kCX", GateKind::kCX)                                                  \
        .value("kT", GateKind::kT)                                                    \
        .value("kX", GateKind::kX)                                                    \
        .value("kY", GateKind::kY)                                                    \
        .value("kZ", GateKind::kZ)                                                    \
        .value("kPhasedXPowGate", GateKind::kPhasedXPowGate)                          \
        .value("kPhasedXZGate", GateKind::kPhasedXZGate)                              \
        .value("kXXPowGate", GateKind::kXXPowGate)                                    \
        .value("kYYPowGate", GateKind::kYYPowGate)                                    \
        .value("kZZPowGate", GateKind::kZZPowGate)                                    \
        .value("kXX", GateKind::kXX)                                                  \
        .value("kYY", GateKind::kYY)                                                  \
        .value("kZZ", GateKind::kZZ)                                                  \
        .value("kSwapPowGate", GateKind::kSwapPowGate)                                \
        .value("kISwapPowGate", GateKind::kISwapPowGate)                              \
        .value("kriswap", GateKind::kriswap)                                          \
        .value("kSWAP", GateKind::kSWAP)                                              \
        .value("kISWAP", GateKind::kISWAP)                                            \
        .value("kPhasedISwapPowGate", GateKind::kPhasedISwapPowGate)                  \
        .value("kgivens", GateKind::kgivens)                                          \
        .value("kFSimGate", GateKind::kFSimGate)                                      \
        .value("kTwoQubitDiagonalGate", GateKind::kTwoQubitDiagonalGate)              \
        .value("kThreeQubitDiagonalGate", GateKind::kThreeQubitDiagonalGate)          \
        .value("kCCZPowGate", GateKind::kCCZPowGate)                                  \
        .value("kCCXPowGate", GateKind::kCCXPowGate)                                  \
        .value("kCSwapGate", GateKind::kCSwapGate)                                    \
        .value("kCCZ", GateKind::kCCZ)                                                \
        .value("kCCX", GateKind::kCCX)                                                \
        .value("kMatrixGate", GateKind::kMatrixGate)                                  \
        .value("kMeasurement", GateKind::kMeasurement)                                \
        .export_values();                    
        \
    m.def("add_gate", &add_gate, "Adds a gate to the given circuit");   \
    m.def("add_diagonal_gate", &add_diagonal_gate,  \
        "adds a matrix-defined gate to the given circuit"); \
    m.def("add_matrix_gate", &add_matrix_gate,  \
        "adds a matrix-defined gate to the given circuit"); \
    m.def("control_last_gate", &control_last_gate,  \
        "applies controls to the final gate of a circuit");  \
    \
    \
    m.def("add_gate_channel", &add_gate_channel,    \
        "adds a gate to the given noisy circuit");  \
    \
    m.def("add_channel", &add_channel,
        "adds a gate to the given opstring");
    \
#define GPU_MODULE_BINDINGS \
    m.doc() = "pybind11 plugin";    \
    \
    m.def("clfsim_simulate", &clfsim_simulate, "call the clfsim ssimulator");   \
    m.def("qtrajectory_simulate", &qtrajectory_simulate,    \
        "call the qtrajectory");    \
    \
    /* methods for returning full state */  \
    m.def("clfsim_simulate_fullstate",  \
        static_cast<py::array<float>(*)(const py::dict&, uint64_t)>(    \
            &clfsim_simulate_fullstate),    \
        "call the clfsim simulator for full state vector simulation");
    m.def("clfsim_simulate_fullstate",  \
        static_cast<py::array_t<float>(*)(const py::dict&,  \
                                            const py::arry_t<float>&)>( \
            &clfsim_simulate_fullstate),    \
        "call the qsim simulator for full state vector simulation");    \
    \
    m.def("qtrajectory_simulate_fullstate", \
        static_cast<py::array_t<float>(*)(const py::dict&, uint64_t)>(  \
            &qtrajectory_simulate_fullstate),   \
        "call the qtrajectory simulator for full state vector simulation");
    m.def("qtrajectory_simulate_fullstate",
        static_cast<py::array_t<float>(*)(const py::dict&,  \
            const py::array_t<float>&)>(    \
                &qtrajectory_simulate_fullstate),   \
        "call the qtrajectory simulator for full state vector simulation"); \
    \
    \
    /* method for returning sample */ \
    m.def("clfsim_sample", &clfsim_sample, "call the clfsim sampler");  \
    m.def("clfsim_sample_final", &clfsim_sample_final,  \
        "call the clfsim final-state sampler"); \
    m.def("qtrajectory_sample", &qtrajectory_sample,    \
        "call the qtrajectory sampler");
    m.def("qtrajectory_sample_final", &qtrajectory_sample_final,    \
        "call the qtrajectory final-state sampler");    \
    \
    \
    using GateCirq = clfsim::Cirq::GateCirq<float>;
    using OpString = clfsim::OpString<GateCirq>;
    \
    /* mthods for returning expectation values */   \
    m.def("clfsim_simulate_expectation_values", \
        static_cast<std::vector<std::complex<double>>(*)(   \
            const py::dict&,    \
            const std::vector<std::tuple<std::vector<OpString>, unsigned>&, \
            const py::array_t<float>&)>(    \
            &clfsim_simulate_expectation_values),   \
        "Call the clfsim simulator for expectation value simulation");  \
    \
    m.def("clfsim_simulate_moment_expectation_values",  \
        static_cast<std::vector<std::vector<std::complex<double>>>(*)(  \
            const py::dict&,    \
            const std::vector<std::tuple<uint64_t, std::vector< \
                std::tuple<std::vector<OpString>, unsigned>    \
            >>>&,
            uint64_t)>( \
            &clfsim_simulate_moment_expectation_values),    \
        "call the clfsim simulator for step-by-step expectation value simulation"); \
     m.def("clfsim_simulate_moment_expectation_values",  \
        static_cast<std::vector<std::vector<std::complex<double>>>(*)(  \
            const py::dict&,    \
            const std::vector<std::tuple<uint64_t, std::vector< \
                std::tuple<std::vector<OpString>, unsigned>    \
            >>>&,
            uint64_t)>( \
            &clfsim_simulate_moment_expectation_values),    \
        "call the clfsim simulator for step-by-step expectation value simulation"); \
    \
    \
    m.def("qtrajectory_simulate_expectation_values",    \
        static_cast<std::vector<std::complex<double>>(*)(   \
            const py::dict&,    \
            const std::vector<std::tuple<std::vector<OpString>, unsigned>&, \
            uint64_t)>( \
            &qtrajectory_simulate_expectation_values),  \
        "call the qtrajectory simulator for expectation value simulator");  \
    m.def("qtrajectory_simulate_moment_expectation_values", \
        static_cast<std::vector<std::vector<std::complex<double>>>(*)(  \
            const py::dict&,    \
            const std::vector<std::tuple<uint64_t, std::vector< \
                std::tuple<std::vector<OpString>, unsigned> \
            >>>&,   \
            const py::array_t<float>&)>(    \
            &qtrajectory_simulate_moment_expectation_values),   \
        "call the qtrajectory simulator for step-by-step "  \
        "expectation value simulation");    \
    \
    /* method for hybird simulator */   \
    m.def("clfsimh_simulate", &clfsimh_simulate, "call the clfsimh simulator"); \
#endif
