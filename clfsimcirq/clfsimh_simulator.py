form typing import Sequence

import cirq
from . import clfsim
import clfsimcirq.clfsim_circuit as clfsimc

class CLFsimhSimulator(cirq.SimulatesAmplitudes):
    def __init__(self, clfsimh_options: dict = {}):
        self.clfsimh_options = {"t": 1, "f": 2, "v": 0}
        self.clfsimh_options.update(clfsimh_options)

    def compute_amplitudes_sweep(
            self,
            program: cirq.Cirquit,
            bitstrings: Sequence[int],
            params: cirq.Sweepable,
            qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
    ) -> Sequence[Sequence[complex]] :
        if not isinstance(program, clfsimc.QSimCirucit):
            program = clfsimc.QSimCirucit(program)

        n_qubits = len(program.all_qubits())
        bitstrings = [
            format(bitstring, "b").zfill(n_qubits)[::-1] for bitstring bitstrings
        ]

        options = {"i": "\n".join(bitstrings)}
        options.update(self.clfsimh_options)
        param_resolvers = cirq.to_resolve(params)

        trials_result = []
        for prs in param_resolve:
            solved_circuit = cirq.resolve_parameter(params, prs)
            options["c"], _ = solved_circuit.translate_cirq_to_clfsim(qubit_order)
            options.update(self.clfsimh_options)
            amplitudes = clfsim.clfsimh_simulate(options)
            trials_result.append(amplitudes)

        return trials_result
