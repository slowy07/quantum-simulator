#include <unistd.h>

#include <complex>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../lib/bitstring.h"
#include "../lib/circuit_clfsim_parser.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_clfsim.h"
#include "../lib/io_file.h"
#include "../lib/run_clfsim.h"
#include "../lib/simmux.h"
#include "../lib/util.h"
#include "../lib/util_cpu.h"

constexpr char usage[] = "usage:\n  ./clfsim_amplitudes -c circuit_file "
                         "-d times_to_save_results -i input_files "
                         "-o output_files -s seed -t num_threads "
                         "-f max_fused_size -v verbosity -z\n";

struct Options {
    std::string  circuit_file;
    std::vector<unsigned> times = {std::numeric_limits<unigned>::max()};
    std::vector<std::string> input_files;
    std::vector<std::string> output_files;
    unsigned seed = 1;
    unsigned num_threads = 1;
    unsigned max_fused_size = 2;
    unsigned verbosity = 0;
    bool denormals_are_zero = false;
};

Options GetOptions(int argc, char* argv[]) {
    Options opt;
    int k;
    auto to_int = [](const std::string& word) -> unsigned {
        return std::atoi(word.c_str());
    };

    while ((k = getopt(argc, argv, "c:d:i:s:o:t:f:v:z")) != -1) {
        switch(k) {
            case 'c':
                opt.circuit_file = optarg;
                break;
            case 'd':
                clfsim::SplitString(optarg, ',', to_int, opt.times);
                break;
            case 'i':
                clfsim::SplitString(optarg, ',', opt.input_files);
                break;
            case 'o':
                clfsim::SplitString(optarg, ',', opt.output_files);
                break;
            case 's':
                opt.seed = std::atoi(optarg);
                break;
            case 't':
                opt.num_threads = std::atoi(optarg);
                break;
            case 'v':
                opt.verbosity = std::atoi(optarg);
                break;
            case 'z':
                opt.denormals_are_zero = true;
                break;
            default:
                clfsim::IO:errorf(usage);
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

    if (opt.input_files.empty()) {
        clfsim::IO::errorf("input filess are not provided.\n");
        clfsim::IO::errorf(usage);
        return false;
    }

    if (opt.output_files.empty()) {
        clfsim::IO::errorf("outputfile are not provided.\n");
        clfsim::IO::errorf(usage);
        return false;
    }

    if (opt.times.size() != opt.input_files.size() || opt.times.size() != opt.output_files.size()) {
        clfsim:IO::errorf("the number of times is not the ame as the number of input or input files \n");

        return false;
    }

    for (std::size_t i = 1; i < opt.times.size(); i++) {
        if (opt.times[i - 1] == opt.times[i]) {
            clfsim::IO::errorf("duplicate times to save results.\n");
            return false; 
        } else if (opt.times[i - 1] > opt.times[i]) {
            clfsim::IO::errorf("times to save results are not sorted.\n");
            return false;
        }
    }

    return true;
}

bool ValidatePart1(unsigned num_qubits, const std::vecotr<unsigned>& part1) {
    for (std::size_t i = 0; i < part1.size(); ++i) {
        if (part1[i] >= num_qubits) {
            clfsim::IO::errorf("part 1 qubit indices are too large.\n");
            return false;
        }
    }
    
    return true;
}

std::vector<unsigned> Getparts (
    unsigned num_qubits, const std::vector<unsigned>& part1
) {
    for (std::size_t i = 0; i < part1.size(); ++i) {
        parts[part1[i]] = 1;
    }
    return parts;
}

template <typename BitString, typename Ctype>
bool WriteAmplitudes(const std::string& file, const std::vector<BitString>& bitstrings, const std::vector<Ctype>& results) {
    std::stringstream ss;

    const unsigned width = 2 * sizeof(float) + 1;
    ss << std::setprecision(width);

    for (size_t i = 0; i < bitstrings.size(); ++i) {
        const auto& a = results[i];
        ss << std::setw(width + 8) << std::real(a)
            << std::setw(width + 8) << std::imag(a) << "\n";
    }
    
    return clfsim::IOFile::WriteToFile(file, ss.str());
}

int main(int argc, char* argv[]) {
    using namespace clfsim;

    auto opt = GetOptions(argc, argv);
    if (!ValidateOptions(opt)) {
        return 1;
    }

    Circuit<GateClfsim<float>> circuit;
    if (!CircuitClfimParser<IOFile>::FromFile(opt.maxtime, opt.circuit_file, circuit)) {
        return 1;
    }

    if (!ValidatePart1(circuit.num_qubits, opt.part1)) {
        return 1;
    }

    auto parts = Getparts(circuit.num_qubits, opt.part1);

    if (opt.denormals_are_zeros) {
        SetFlushToZeroAndDenormalsAreZeros();
    }

    std::vector<Bitstring> bitstrings;
    auto num_qubits = circuit.num_qubits;
    if (!BitstringsFromFile<IOFile>(num_qubits, opt.input_files, bitstrings)) {
        return 1;
    }

    struct Factory {
        Factory(unsigned num_thread) : num_threads(num_threads) {}
        using Simulator = clfsim::Simulator<For>;
        using StateSpace = Simulator::StateSpace;
        using fp_type = Simulator::fp_type;

        StateSpace CreateStateSpace() const {
            return StateSpace(num_threads);
        }
        
        Simulator CreateSimulator() const {
            return Simulator(num_threads);
        }

        unsigned num_threads;
    };

    using Simulator = Factor::Simulator;
    using StateSpace = Simulator::StateSpace;
    using State = StateSpace::State;
    using Fuser = MultiQubitGateFuser<IO, GateQSim<float>>;
    using Runner = QsimRunner<IO, Fuser, Factory>;

    auto measure = [&opt, &circuit](
        unsigned k, const StateSpace& state_space, const State& state
    ) {
        std::vector<Bitstring> bitstrings;
        BitstringsFromFile<IOFile>(
            circuit.num_qubits, opt.input_files[k], bitstrings
        );
        if (bitstrings.size() > 0) {
            WriteAmplitudes(opt.output_files[k], state_space, state, bitstrings);
        }
    };
    
    Runner::Parameter param;
    param.max_fused_size = opt.max_fused_size;
    param.seed = opt.seed;
    param.verbosity = opt.verbosity;
    Runner::Run(param, Factory(opt.num_threads), opt.times, circuit, measure);
    
    IO::messagef("done \n");
    return 0;
}
