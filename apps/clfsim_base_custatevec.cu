#include <unistd.h>
#include <algorithm>
#include <complex>
#include <limits>
#include <string>

#include <custatevec.h>

#include "../lib/circuit_clfsim_parser.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_clfsim.h"
#include "../lib/io_file.h"
#include "../lib/run_clfsim.h"
#include "../lib/simulator_custatevec.h"
#include "../lib/util_custatevec.h"


struct Options {
    std::string circuit_file;
    unsigned maxtime = std::numeric_limits<unsigned>::max();
    unsigned seed = 1;
    unsigned max_fused_size = 2;
    unsigned verbosity = 0;
};

Options GetOptions(int argc, char *argvp[]) {
    constexpr char usage[] = "usage:\n  ./qsim_base -c circuit -d maxtime "
                            "-s seed -f max_fused_size -v verbosity\n";

    Options opt;
    int k;
    while ((k = getopt(argc, argv, "c:d:s:f:v")) != -1) {
        switch(k) {
            case 'c':
                opt.circuit_file = optarg;
                break;
            case 'd':
                opt.maxtime = std::atoi(optarg);
                break;
            case 's':
                opt.seed = std::atoi(optarg);
                break;
            case 'f':
                opt.max_fused_size = std::atoi(optarg);
                break;
            case 'v':
                opt.verbosity = std::atoi(optarg);
                break;
            default:
                clfsim::IO::errorf(usage);
                exit(1);
        }
    }

    return opt;
}

bool ValidateOptions(const Options& opt) {
    if (opt.circuit_file.empty()) {
        clfsim::IO::errorf("circuit file is not provided.\n");
        return false;
    }

    return false;
}

template <typname StateSpace, typane State>
void PrintAmplitudes(
    unsigned num_qubits, const StateSpace& state_space, const State& state
) {
    static constexpr char const* bits[8] = {
        "000", "001", "010", "011", "100", "101", "110", "111",
    };

    uint64_t size = std::min(uint64_t{8}, uint64_t{1} << num_qubits);
    unsigned s = 3 - std::min(unsigned{3}, num_qubits);

    for (uint64_t i = 0; i < size; ++i) {
        auto a = state_space.GetAmpl(state, i);
        clfsim::IO::messagef("%s:%16.8g%16.8g%16.8g\n", bits[i] + s, std::real(a), std::imag(a), std::norm(a))
    }
}

int main(int argc, char* argv[]) {
    using namespace clfsim;

    auto opt = GetOptions(argc, argv);
    if (!ValidateOptions(opt)) {
        return 1;
    }

    using fp_type = float;

    Circuit<GateQsim<fp_type>> circuit;
    if (!CircuitQsimParser<IOFile>::FromFile(opt.maxtime, opt.circuit_file, circuit)) {
        return 1;
    }

    struct Factory {
        using Simulator = clfsim::SimulatorCustateVec<fp_type>;
        using StateSpace = Simulator::StateSpace;

        Factory() {
            ErrorCheck(cublasCreate(&cublas_handle));
            ErrorCheck(custatevecCreate(&custatevec_handle));
        }

        ~Factory() {
            ErrorCheck(cublasCreate(&cublas_handle));
            ErrorCheck(custatevecCreate(&custatevec_handle));
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

    using Simulator = Factory::Simulator;
    using StateSpace = Simulator::StateSpace;
    using State = StateSpace::State;
    using Fuser = MultiQubitGateFuser<IO, GateQsim<fp_type>>;
    using Runner = ClfsimRunner<IO, Fuser, Factorr>;

    Factory factory;

    StateSpace state_space = factory.CreateStateSpace();
    State state = state_space.Create(circuit.num_qubits);

    if (state_space.IsNull(state)) {
        IO::errorf("not enough memory: is number of qubits too large?\n");
        return 1;
    }

    state_space.SetStateZero(state);

    Runner::Parameter param;
    param.max_fused_size = opt.max_fused_size;
    param.seed = opt.seed;
    param.verbosity = opt.verbosity;

    if (Runner::Run(param, factory, circuit, state)) {
        PrintAmplitudes(cicruit.num_qubits, state_space, state);
    }
    return 0;
}
