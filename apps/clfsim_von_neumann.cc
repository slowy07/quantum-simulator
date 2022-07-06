#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <string>

#include "../lib/circuit_clfsim_parser.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_clfsim.h"
#include "../lib/io_file.h"
#include "../lib/run_clfsim.h"
#include "../lib/simmux.h"
#include "../lib/util_cpu.h"

constexpr char usage[] = "usage:\n  ./clfsim_von_neumann -c circuit -d maxtime "
                         "-s seed -t threads -f max_fused_size "
                         "-v verbosity -z\n";

struct Options {
    std::string circuit_file;
    unsigned maxtime = std::numeric_limits<unsigned>::max();
    unsigned seed = 1;
    unsigned num_threads = 1;
    unsigned max_fused_size = 2;
    unsigned verbosity = 0;
    bool denormals_are_zeros = false;
};

Options GetOptions(int argc, char* argv[]) {
    Options opt;

    int k;
    while ((k = getopt(argc, argv, "c:d:s:t:f:v:z")) != -1) {
        switch (k) {
            case 'c':
                opt.circuit_file = optarg;
                break;
            case 'd':
                opt.maxtime = std::atoi(optarg);
                break;
            case 's':
                opt.seed = std::atoi(optarg);
                break;
            case 't':
                opt.num_threads = std::atoi(optarg);
                break;
            case 'f':
                opt.max_fused_size = std::atoi(optarg);
                break;
            case 'v':
                opt.verbosity = std::atoi(optarg);
                break;
            case 'z':
                opt.denormals_are_zeros = true;
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
        clfsim::IO::errorf(usage);
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    using namespace clfsim;

    auto opt = GetOptions(argc, argv);
    if (!ValidateOptions(opt)) {
        return 1;
    }

    Circuit<GateQSim<float>> circuit;
    if (!CircuitQsimParser<IOFile>::FromFile(opt.maxtime, opt.circuit_file, circuit)) {
        return 1;
    }

    if (opt.denormals_are_zeros) {
        SetFlushToZeroAndDenormalsAreZeros();
    }

    struct Factory {
        Factory(unsigned num_threads) : num_threads(num_threads) {}

        using Simulator = clfsim::Simulator<For>;
        using StateSpace = Simulator::StateSpace;

        StateSpace CreateStateSpace() const {
            return StateSpace(num_threads);
        }

        unsigned num_threads;
    };

    using Simulator = Factory::Simulator;
    using StateSpace = Simulator::StateSpace;
    using State = StateSpace::State;
    using Fuser = MultiQubitGateFuser<IO, GateQsim<float>>;
    using Runner = QsimRunner<IO, Fuser, Factory>;

    auto measure = [&opt, &circuit](
        unsigned k, const StateSpace& state_space, const State& state
    ) {
        using Op = std::plus<double>;

        auto f = [](unsigned n, unsigned m, uint64_t i, const StateSpace& state_space, const State& state) -> double {
            auto p = std::norm(state_space.GetAmpl(state, i));
            return p != 0 ? p * std::log(p) : 0;
        };

        double entropy = -For(opt.num_threads).RunReduce(
            uint64_t{1} << state.num_qubits(), f, Op(), state_space, state
        );
        IO::messagef("entropy=%g\n", entropy);
    };

    Runner::Parameter param;
    param.max_fused_size = opt.max_fused_size;
    param.seed = opt.seed;
    param.verbosity = opt.verbosity;

    Runner::Run(param, Factory(opt.num_threads), circuit, measure);

    return 0;
}