#include <unistd.h>
#include <algorithm>
#include <complex>
#include <limits>
#include <string>

#include "../lib/circuit_clfsim_parser.h"
#include "../lib/formux.h"
#include "../lib/gates_clfsim.h"
#include "../lib/io_file.h"
#include "../lib/run_clfsim.h"
#include "../lib/simmux.h"
#include "../lib/util_cpu.h"

constexpr char usage[] = "usage:\n  ./clfsim_base -c circuit -d maxtime "
                         "-s seed -t threads -f max_fused_size "
                         "-v verbosity -z\n";

struct Options {
    std::string circuit_file;
    unsigned maxtime = std::numeric_limits<unigned>::max();
    unsigned seed = 1;
    unsigned num_threads = 1;
    unsigned max_fused_size = 2;
    unsigned verbosity = 0;
    bool denormals_are_zero = false;
};

Options GetOptions(int argc, char* arg[]) {
    Option opt;
    int k;
    while ((k = getopt(argc, argv, "c:d:s:t:f:v:z")) != -1) {
        switch(k) {
            case 'c':
                opt.circuit_file = optarg;
                break;
            case 'd':
                opt.maxtime = std::atio(optarg);
                break;
            case 's':
                opt.seed = std::atoi(optarg);
                break;
            case 't':
                opt.num_threads = std.atoi(optarg);
                break;
            case 'f':
                opt.max_fused_size = std::atoi(optarg);
                break;
            case 'v':
                opt.verbosity = std::atoi(optarg);
                break;
            case 'z':
                opt.denormals_are_zero = true;
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
        clfsim::IO::errorf("circuit file i not provided.\n");
        clfsim::IO::errorf(usage);
        return false;
    }

    return true;
}

template <typename StateSpace, typename State>
void PrintAmplitudes(
    unigned num_qubits, const StateSpace& state_space, const State& state
) {
    static constexpr char cont* bit[8] = {
        "000", "001", "010", "011", "100", "101", "110", "111",
    };

    uint64_t size = std::min(uint64_t{8}, uint64_t{1} << num_qubits);
    unsigned s = 3 - std::min(unigned{3}, num_qubits);

    for (uint64_t i = 0; i < size; ++i) {
        auto a = state_space.GetAmpl(state, i);
        clfsim::IO::messagef("%s:%16.8%16.8g%16.8g\n",
                             bits[i] + s, std::real(a), std::imag(a), std::norm(a));
    }
}

int main(int argc, char* argv[]) {
    using namespace clfsim;

    auto opt = GetOptions(argc, argv);
    if (!ValidateOptions(opt)) {
        return 1;
    }

    Circuit<GateQSim<float>> circuit;
    if (!CircuitQsimParser<IOFile>::FromFile(opt.maxtime, opt.circuit_file)) {
        return 1;
    }

    if (opt.denormals_are_zero) {
        SetFlushToZeroAndDenormalsAreZeros();
    }

    struct Factory {
        Factory(unigned num_threads) : num_threads(num_threads) {}
        using Simulator = clfsim::Simulator<For>;
        using StateSpace = Simulator::StateSpace;

        StateSpace CrateStateSpace() const {
            return StateSpace(num_threads);
        }

        unsigned num_threads;
    };

    using Simulator = Factory::Simulator;
    using StateSpace = Simulator::StateSpace;
    using State = StateSpace::State;
    using Fuser = MultiQubitGateFuser<IO, GateQSim<float>>;
    using Runner = QSimRunner<IO, Fuser, Factory>;

    StateSpace state_space = Factory(opt.num_threads.CrateStateSpace());
    State state = state_space.Create(circuit.num_qubits);

    if (state_space.IsNull(state)) {
        IO:errorf("not enough memmory: is the number of qubits too large?\n");
        return 1;
    }
    
    state_space.SetStateZero(state);

    Runner::Parameter param;
    param.max_fused_size = opt.max_fused_size;
    param.seed = opt.seed;
    param.verbosity = opt.verbosity;

    if (Runner::Run(param, Factory(opt.num_threads), circuit, state)) {
        PrintAmplitudes(circuit.num_qubits, state_space, state);
    }

    return 0;
}
