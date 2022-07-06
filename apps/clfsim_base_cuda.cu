#include <unistd.h>
#include <algorithm>
#include <complex>
#include <limits>
#include <string>

#include "../lib/circuit_clfsim_parser.h"
#include "../lib/formux.h"
#include "../lib/fuser.h"
#include "../lib/gates_clfsim.h"
#include "../lib/io_file.h"
#include "../lib/run_clfsim.h"
#include "../lib/simulator_cuda.h"


struct Options {
    std::string circuit_file;
    unsigned maxtime = std::numeric_limits<unsigned>::max();
    unsigned seed = 1;
    unsigned max_fused_size = 2;
    unsigned num_threads = 256;
    unsigned num_dblocks = 16;
    unsigned verbosity = 0;
};

Options GetOptions(int argc, char* argv[]) {
    constexpr char usage[] = "usage:\n  ./clfsim_base -c circuit -d maxtime "
                           "-s seed -f max_fused_size -t num_threads"
                           "-n num_dblocks -v verbosity\n";

    Options opt;
    int k;
    while ((k = getopt(argc, argv, "c:d:s:f:t:n:v:")) != -1) {
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
            case 't':
                opt.num_threads = std::atoi(optarg);
                break;
            case 'n':
                opt.num_dblocks = std::atoi(optarg);
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
        clfsim::IO::errorf("circuit file is not provided \n")
        return false;
    }
    return true;
}

template <typename StateSpace, typename State>
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
        clfsim::IO::messagef("%s:%16.8g%16.8g%16.8g\n", bits[i] + s, std::real(a), std::imag(a), std::norm(a));
    }
}

int main(int argc, char* argv[]) {
    using namespace clfsim;

    auto opt = GetOptions(argc, argv);
    if (!ValidateOptions(opt)) {
        return 1;
    }

    circuit<GateQsim<float>> circuit;
    if (!CircuitQsimParser<IOfile>::FromFile(opt.maxtime, opt.cicruit_file, cicruit)) {
        return 1;
    }

    struct Factory{
        using Simulator = clfsim::SimultorCUDA<float>;
        using StateSpace = Simulator::StateSpace;

        Factory(
            const StateSpace::Parameter& param1,
            const Simulator::Parameter& param2
        ) : param1(param1), param2(param2) {}

        StateSpace CreateStateSpace() const {
            return StateSpace(param1);
        }

        Simulator CrateSimulator() const {
            return Simulator(param2);
        }

        const StateSpace::Parameter& param1;
        const Simulator::Parameter& param2;
    };

    using Simulator = Factory::Simulator;
    using StateSpace = Simulator::StateSpace;
    using State = StateSpace::State;
    using Fuser = MultiQubitGateFuser<IO, GateQsim<float>>;
    using Runner = QsimRunner<IO, Fuser, Factory>;

    StateSpace::parameter param1;
    param1.num_threads = opt.num_threads;
    param1.num_dblocks = opt.num_dblocks;

    Simulator::parameter param2;
    param2.num_threads = opt.num_threads;
    
    Factory factory(param1, param2);

    StateSpace state_space = factory.CreateStateSpace();
    State state = state_space.Create(circuit.num_qubits);

    if (state_space.isNull(state)) {
        IO::errorf("not enough memmory: the number of qubits its to large \n")
        return 1;
    }

    state_space.SetStateZero(state);

    Runner::Parameter param3;
    param3.max_fused_size = opt.max_fused_size;
    param3.seed = opt.seed;
    param3.verbosity = opt.verbosity;

    if (Runner::Run(param3, factory, circuit, state)) {
        PrintAmplitudes(circuit.num_qubits, state_space, state);
    }
    return 0;
}