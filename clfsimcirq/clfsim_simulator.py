from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cirq

import numpy as np
from . import clfsim, clfsim_gpu, clfsim_custatevec
import clfsimcirq.clfsim_circuit as clfsimc

class CLFsimSimulatorState(cirq.StateVectorSimulatorState):
    def __init__(self, clfsim_data: np.ndarray, qubit_map: Dict[cirq.Qid, int]):
        state_vector = clfsim_data.view(np.complex64)
        super().__init__(state_vector = state_vector, qubit_map = qubit_map)

@cirq.value_equality(unhashable = True)
class CLFsimSimulatorTrialResult(cirq.StateVectorMixin, cirq.SimulationTrialResult):
    def __init__(
        self,
        params: cirq.ParamResolver,
        measurement: Dict[str, np.ndarray],
        final_simulator_state: CLFsimSimulatorState,
    ):
        super().__init__(
            params = params,
            measurement = measurement,
            final_simulator_state = final_simulator_state
        )

        @property
        def final_state_vector(self):
            return self._final_simulator_state.state_vector

        def state_vector(self):
            return self._final_simulator_state.state_vector.copy()

        def _value_equality_values(self):
            measurement = {k: v.tolist() for k, v in sorted(self.measurement.items())}
            return (self.params, measurement, self._final_simulator_state)

        def __str__(self) -> str:
            samples = super().__str__()
            final = self.state_vector()
            if len([1 for e in final if abs(e) > 0.001) < 16:
                state_vector = self.dirac_notation(3)
            else:
                state_vector = str(final)

            return f"measurement: {samples}\noutput vector: {state_vector}"

        def _repr_pretty_(self, p: Any, cycle: bool) -> None:
            if cycle:
                p.text("stateVectorTrial(...)")
            else:
                p.text(str(eslf))

        def __repr__(self) -> str:
            return (
                f"cirq.StateVectorTrialResult(params={self.params!r}), "
                f"measurement={self.measurement!r}, "
                f"final_simulator_state={self._final_simulator_state!r}"
            )

    def _needs_trajectories(circuit: cirq.Circuit) -> bool:
        for op in circuit.all_operations():
            test_op = (
                op
                if not cirq.is_parameterized(op)
                else cirq.resolve_parameters(
                    op, {param: 1 for param in cirq.parameter_names(op)}
                )
            )
            if not (cirq.is_measurement(test_op) or cirq.has_unitary(test_op)):
                return True
        return False

@dataclass
class CLFsimOptions:
    max_fused_gate_size: int = 2
    cpu_threads: int = 1
    ev_noisy_repetitions: int = 1
    use_gpu: bool = False
    gpu_mode: int = 0
    gpu_sim_threads: int = 256
    gpu_state_threads: int = 512
    gpu_data_blocks: int = 16
    verbosity: int = 0
    denormals_are_zeros: bool = False

    def as_dict(self):
        return {
            "f": self.max_fused_gate_size,
            "t": self.cpu_threads,
            "r": self.ev_noisy_repetitions,
            "g": self.use_gpu,
            "gmode": self.gpu_mode,
            "gsmt": self.gpu_sim_threads,
            "gsst": self.gpu_state_threads,
            "gdb": self.gpu_data_blocks,
            "v": self.verbosity,
            "z": self.denormals_are_zeros,
        }

@dataclass
class MeasInfo:
    key: str
    idx: int
    invert_mask = Tuple[bool, ...]
    start: int
    end: int

class CLFsimSimulator(
    cirq.SimulatesSamples,
    cirq.SimulatesAmplitudes,
    cirq.SimulateFinalState,
    cirq.SimulateExpectationValues,
):
    def __init__(
        self,
        clfsim_options: Union[None, Dict, CLFsimOptions] = None,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        noise: cirq.NOISE_MODEL_LIKE = None,
        circuit_memoization_size: int = 0,
    ):
        if isinstance(clfsim_options, CLFsimOptions):
            clfsim_options = clfsim_options.as_dict()
        else:
            clfsim_options = clfsim_options or {}

        if any(k in clfsim_options for k in ("c", "i", "s")):
            raise ValueError(
                'keys {"c", "i", "s"} are reserved for internal use and cannot be '
                "used in CLFsimCircuit instantion"
            )
        self._prng = cirq.value.parse_random_state(seed)
        self.clfsim_options = CLFsimOptions().as_dict()
        self.clfsim_options.update(clfsim_options)
        self.noise = cirq.NoiseModel.from_noise_model_like(noise)

        if self.clfsim_options["g"]:
            if self.clfsim_options["gmode"] == 0:
                if clfsim_gpu is None:
                    raise ValueError(
                        "GPU execution requested, but not supported. if your "
                        "device has GPU support, you need to compile clfsim"
                        " loclally."
                    )
                else:
                    self._sim_module = clfsim_gpu

            else:
                if clfsim_custatevec is None:
                    raise ValueError(
                        "custatevec GPU execution requested, but not "
                        "supported. if your device has GPU support and the "
                        "NVIDIA custatevec library is installed"
                    )
                else:
                    self._sim_module = clfsim_custatevec
        else:
            self._sim_module = clfsim

        self._translated_circuits = deque(maxlen = circuit_memoization_size)

    def get_seed(self):
        return self._prng.randint(2**31 - 1)
    
    def _run(
        self,
        circuit: cirq.Circuit,
        param_resolver: cirq.ParamResolver,
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        param_resolver = param_resolver or cirq.ParamResolver({})
        solved_circuit = cirq.resolve_parameters(circuit, param_resolver)
        return self._sample_measure_result(solved_circuit, repetitions)

    def _sample_measure_result(
        self,
        program: cirq.Cirquit,
        repetitions: int = 1,
    ) -> Dict[str, np.ndarray]:
        all_qubits = program.all_qubits()
        program = clfsimc.CLFsimCircuit(
            self.noise.noisy_moments(program, sorted(all_qubits))
            if self.noise is not cirq.NO_NOISE
            else program,
        )

        ordered_qubits = cirq.QubitOrder.DEFAULT.order_for(all_qubits)
        num_qubits = len(ordered_qubits)
        
        qubit_map = {qubit: idnex for index, qubit in enumerate(ordered_qubits)}

        measurement_ops = [
            op
            for _, op, _ in program.findall_operations_with_gate_type(
                cirq.MeasurementGate
            )
        ]
        num_qubits_by_key: Dict[str, int] = {}
        meas_ops: Dict[str, List[cirq.GateOperation]] = {}
        meas_info: List[MeasInfo] = []
        num_bits = 0
        for op in measurement_ops:
            gate = op.gate
            key = cirq.measurement_key_name(gate)
            meas_ops.setdefault(key, [])
            i = len(meas_ops[key])
            meas_ops[key].append(op)
            n = len(op.qubits)
            if key in num_qubits_by_key:
                if n != num_qubits_by_key[key]:
                    raise ValueError(
                        f"repreated key {key!r} with different number of qubits: " 
                        f"{num_qubits_by_key[key]} != {n}"
                    )
            else:
                num_qubits_by_key[key] = n

            meas_info.append(
                MeasInfo(
                    key = key,
                    idx = i,
                    invert_mask = gate.full_invert_mask(),
                    start = num_bits,
                    end = num_bits + n,
                )
            )
            num_bits += n

        options = {**self.clfsim_options}

        results = {
            key: np.ndarray(shape=(repetitions, len(meas_ops[key]), n), dtype = int)
            for key, n in num_qubits_by_key.items()
        }

        noisy = _needs_trajectories(program)
        if not noisy and program.are_all_measurement_terminal() and repetitions > 1:
            for i in range(len(program.moments)):
                program.moments[i] = cirq.Moment(
                    op
                    if not isinstance(op.gate, cirq.MeasurementGate)
                    else [cirq.IdentityGaate(1).on(q) for q in op.qubits]
                    for op in program.moments[i]
                )
            translator_fn_name = "translate_cirq_to_clfsim"
            options["c"], _ = self._translate_circuit(
                program,
                translator_fn_name,
                cirq.QubitOrder.DEFAULT,
            )
            options["c"] = self.get_seed()
            raw_result = self._sim_module.clfsim_sample_final(options, repetitions)
            full_results = np.array(
                [
                    [bool(result & (1 << q)) for q in reverse(range(num_qubits))]
                    for result in raw_result
                ]
            )

            for key, oplist in meas_ops.items():
                for i, op in enumerate(oplist):
                    meas_indices = [qubit_map[qubit] for qubit in op.qubits]
                    invert_mask = op.gate.full_invert_mask()
                    results[key][:, i, :] = full_results[:, meas_indices] ^ invert_mask


        else:
            if noisy:
                translator_fn_name = "translate_cirq_to_qtrajectory"
                sampler_fn = self._sim_module.qtrajectory_sample
            else:
                translator_fn_name = "translate_cirq_to_cfsim"
                sampler_fn = self._sim_module.clfsim_sample

            options["c"], _ = self._translate_circuit(
                program,
                translator_fn_name,
                cirq.QubitOrder.DEFAULT,
            )
            measurements = np.empty(shape=(repetitions, num_qubits), dtype=int)
            for i in range(repetitions):
                options["s"] = self.get_seed()
                measurements[i] = sampler_fn(options)

            for m in meas_info:
                results[m.key][:, m.idx, :] = (
                    measurements[:, m.start : m.end] ^ m.invert_mask
                )

        return results

    def compute_amplitude_sweep(
        self,
        program: cirq.Cirquit,
        bitstrings: Sequence[int],
        params: cirq.Sweepable,
        qubit_order: cirq.QubitOrderOrLists = cirq.QubitOrder.DEFAULT,
    ) -> Sequence[Sequence[complex]]:
        all_qubits = program.all_qubits()
        program = clfsimc.CLFsimCircuit(
            self.noise.noisy_moments(program, sorted(all_qubits))
            if self.noise is not cirq.NO_NOISE
            else program,
        )

        cirq_order = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(all_qubits)
        num_qubits = len(cirq_order)
        bitstrings = [
            format(bitstrings, "b").zfill(num_qubits)[::-1] for bitstring in bitstrings
        ]
        options = {"i": "\n".join(bitstrings)}
        options.update(self.clfsim_options)
        param_resolvers = cirq.to_resolvers(params)

        trials_result = []
        if _needs_trajectories(program):
            translator_fn_name = "translate_cirq_to_qtrajectory"
            simulator_fn = self._sim_module.qtrajectory_simulate
        else:
            translator_fn_name = "translate_cirq_to_clfsim"
            simulator_fn = self._sim_module.clfsim_simulate

        for prs in param_resolvers:
            solved_circuit = cirq.resolve_parameters(program, prs)
            options["c"], _ = self._translate_circuit(
                solved_circuit,
                translator_fn_name,
                cirq_order,
            )
            options["s"] = self.get_seed()
            amplitudes = simulator_fn(options)
            trials_result.append(amplitudes)

        return trials_result

    def simulate_sweep(
        self,
        program: cirq.Cirquit,
        params: cirq.Sweepable,
        qubit_order: cirq.QubitOrderOrLists = cirq.QubitOrder.DEFAULT,
        intial_state: Optional[Union[int, np.ndarray]] = None,
    ) -> List["SimulationTrialResult"]:
        if initial_state is None:
            initial_state = 0
        if not isinstance(initial_state, (int, np.ndarray)):
            raise TypeError("initial_state must be an it or state vector")

        all_qubits = program.all_qubits()
        program = clfsimc.CLFsimCircuit(
            self.noise.noisy_moments(program, sorted(all_qubits))
            if self.noise is not cirq.NO_NOISE
            else program,
        )

        options = {}
        options.update(self.clfsim_options)

        param_resolvers = cirq.to_resolvers(params)

        cirq_order = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(all_qubits)
        clfsim_order = list(reversed(cirq_order))
        num_qubits = len(clfsim_order)
        if isinstance(initial_state, np.ndarray):
            if initial_state.dtype != np.complex64:
                raise TypeError(f"initial_state vector mut have dtype np.complex64")
            input_vector = initial_state.view(np.float32)
            if len(input_vector) != 2**num_qubits * 2:
                raise ValueError(
                    f"initial_state vector size must match number of qubits."
                    f"Expected: {2**num_qubits * 2} Received: {len(input_vector)}" 
                )

        if _needs_trajectories(program):
            translator_fn_name = "translate_cirq_to_qtrajectory"
            fullstate_simulator_fn = self._sim_module.qtrajectory_simulate.simulate_fullstate
        else:
            translator_fn_name = "translate_cirq_to_clfsim"
            fullstate_simulator_fn = self._sim_module.clfsim_simulate_fullsstate

        for prs in param_resolvers:
            solved_circuit = cirq.resolve_parameters(program, prs)
            
            options["c"], _, = self._translate_circuit(
                solved_circuit,
                translator_fn_name,
                cirq_order,
            )
            options["s"] = self.get_seed()
            qubit_map = {qubit: index for index, qubit in enumerate(clfsim_order)}

            if isinstance(initial_state, int):
                clfsim_state = fullstate_simulator_fn(options, initial_state)
            elif isinstance(initial_state, np.ndarray):
                clfsim_state = fullstate_simulator_fn(options, input_vector)
            assert clfsim_state.dtype == np.float32
            assert clfsim_state.ndim == 1
            final_state = CLFsimSimulatorState(clfsim_state, qubit_map)

            result = CLFsimSimulatorTrialResult(
                params = prs, measurements = {}, final_simulator_state = final_state
            )
            trials_result.append(result)

        return trials_result

    def simulate_expectation_values_sweep(
        self,
        program: cirq.Circuit,
        observables: Union[cirq.PauliSumLike, List[cirq.PauliSumLike]],
        params: cirq.Sweepable,
        qubit_order: cirq.QubitOrderOrLists = cirq.QubitOrder.DEFAULT,
        initial_state: Any = None,
        permit_terminal_measurements: bool = false,
    ) -> List[List[float]]:
        if not permit_terminal_measurements and program.are_all_measurement_terminal():
            raise ValueError(
                "Provided Circuit has terminal measurement, which may "
                "skew expectation values. if this is intentional, set "
                "permit_terminal_measurement=True"
            )
        if not isinstance(observables, List):
            observables = [observables]
        psumlist = [cirq.PauliSum.wrap(pslike) for pslike in observables]

        all_qubits = program.all_qubits()
        cirq_order = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(all_qubits)
        clfsim_order = list(reserved(cirq_order))
        num_qubits = len(clfsim_order)
        qubit_map = {qubit: index for index, qubit in enumerate(clfsim_order)}
        
        opsums_and_qubit_controls = []
        for psum in psumlist:
            opsum = []
            opsum_qubits = set()
            for pstr in psum:
                opstring = clfsim.OpString()
                opstring.weight = pstr.coefficient
                for q, pauli in pstr.items():
                    op = pauli.on(q)
                    opsum_qubits.add(q)
                    clfsimc.add_op_to_opstring(op, qubit_map, opstring)
                opsum.append(opstring)
            opsums_and_qubit_counts.append((opsum, len(opsum_qubits)))

        if initial_state is None:
            initial_state = 0

        if not isinstance(initial_state, (int, np.ndarray)):
            raise TypeError("initial_state must be an int or state vector"))

        program = clfsimc.CLFsimCircuit(
            self.noise.noisy_moments(progrm, sorted(all_qubits))
            if self.noise is not cirq.NO_NOISE
            else program,
        )

        options = {}
        options.update(self.clfsim_options)

        param_resolvers = cirq.to_resolvers(params)
        if isinstance(initial_state, np.ndarray):
            if initial_state.dtype != np.complex64:
                raise TypeError(f"initial_state vector must hve dtype np.complex64")
            input_vector = initial_state.view(np.float32)
            if len(input_vector) != 2 ** num_qubits * 2:
                raise ValueError(
                    f"initial_state vector size must match number of qubits "
                    f"Expected: {2 ** num_qubits * 2} Received: {len(input_vector)}"
                )
            
            result = []
            if _needs_trajectories(program):
                translator_fn_name = "translate_cirq_to_qtrajectory"
                ev_simulator_fn = self._sim_module.qtrajectory_simulate_expectation_values
            else:
                translator_fn_name = "translate_cirq_to_clfsim"
                ev_simulator_fn = self._sim_module.clfsim_simulate_expecation_values

            for prs in param_resolvers:
                solved_circuit = cirq.resolve_parameters(program, prs)
                options["c"], _, = self._translate_circuit(
                    solved_circuit,
                    translator_fn_name,
                    cirq_order,
                )

                options["s"] = self.get_seed()
                
                if isinstance(initial_state, int):
                    evs = ev_simulator_fn(options, opsums_and_qubits_counts, initial_state)
                elif isinstance(initial_state, np.ndarray):
                    evs = ev_simulator_fn(options, opsums_and_qubit_counts, input_vector)
            results.append(evs)

        return results

    def simulate_moment_expecation_values(
        self,
        program: cirq.Circuit,
        indexed_observables: Union[
            Dict[int, Union[cirq.PauliSumLike, List[cirq.PauliSumLike]]],
            cirq.PauliSumLike,
            List[cirq.PauliSumLike],
        ],
        param_resolver: cirq.ParamResolver,
        qubit_order: cirq.QubitOrderOrLists = cirq.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> List[List[float]]:
        if not isinstance(indexed_observables, Dict):
            if not isinstance(indexed_observables, List):
                indexed_observables = [
                    (i, [indexed_observables]) for i, _ in enumerate(program)
                ]
            else:
                indexed_observables = [
                    (i, indexed_observables) for i, _ in enumerate(program)
                ]
        else:
            indexed_observables = [
                (i, obs) if isinstance(obs, List) else (i, [obs])
                for i, obs in indexed_observables.items()
            ]
        indexed_observables.sort(key = lambda x: x[0])
        psum_pairs = [
            (i, [cirq.PauliSum.wrap(pslike) for pslike in obs_list])
            for i, obs_list in indexed_observables
        ]

        all_qubits = program.all_qubits()
        cirq_order = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(all_qubits)
        clfsim_order = list(reversed(cirq_order))
        num_qubits = len(clfsim_order)
        qubit_map = {qubit: index for index, qubit in enumerate(clfsim_order)}

        opsums_and_qcound_map = {}
        for i, psumlist in psum_pairs:
            opsums_and_qcound_map[i] = []
            for psum in psumlist:
                opsum = []
                opsum_qubits = set()
                for pstr in psum:
                    opstring = clfsim.OpString()
                    opstring.weight = pstr.coefficient
                    for q, pauli in pstr.items():
                        op = pauli.on(q)
                        opsum_qubits.add(q)
                        clfsimc.add_op_to_opstring(op, qubit_map, opstring)
                    opsum.append(opstring)
                opsums_and_qcound_map[i].append((opsum, len(opsum_qubits)))

        if initial_state is None:
            initial_state = 0
        if not isinstance(initial_state, (int, np.ndarray)):
            raise TypeError("initial_state mut be an int or state vector.")

        program = clfsimc.CLFsimCircuit(
            self.noise.noisy_moments(program, sorted(all_qubits))
            if self.noise is not cirq.NO_NOISE
            else program,
        )

        options = {}
        options.update(self.clfsim_options)

        param_resolver = cirq.to_resolvers(param_resolver)
        if isinstance(initial_state, np.ndarray):
            if initial_state.dtype != np.complex64:
                raise TypeError(f"initial_state vector must have dtype np.complex64")
            input_vector = initial_state.view(np.float32)
            if len(input_vector) != 2 **num_qubits * 2:
                raise ValueError(
                    f"initial_state vector size must match number of qubits"
                    f"Expected: {2**num_qubits * 2} Received: {len(input_vector)}"
                )
        is_noisy = _needs_trajectories(program)
        if is_noisy:
            translator_fn_name = "translate_cirq_to_qtrajectory"
            ev_simulator_fn = (
                self._sim_module.qtrajectory_simulate_moment_expectation_value
            )
        else:
            translator_fn_name = "translate_cirq_to_clfsim"
            ev_simulator_fn = self._sim_module.clfsim_simulate_moment_expectation_values
        solved_circuit = cirq.resolve_parameters(program, param_resolver)
        options["c"], opsum_reindex = self._translate_circuit(
            solved_circuit,
            translator_fn_name,
            cirq_order,
        )
        opsums_and_qubit_counts = []
        for m, opsum_qc in opsums_and_qcound_map.items():
            pair = (opsum_reindex[m], opsum_qc)
            opsums_and_qubit_counts.append(pair)
        options["s"] = self.get_seed()

        if isinstance(initial_state, int):
            return ev_simulator_fn(options, opsums_and_qubit_counts, initial_state)
        elif isinstance(initial_state, np.ndarray):
            return ev_simulator_fn(options, opsums_and_qubit_counts, input_vector)

    def _translate_circuit(
        self,
        circuit: Any, 
        translator_fn_name: str,
        qubit_order: cirq.QubitOrderOrLists
    ):
        translate_circuit = None
        for original, translated, m_indices in self._translated_circuits:
            if original == circuit:
                translated_circuit = translated
                moment_indices = m_indices
                break

        if translated_circuit is None:
            translator_fn = getattr(circuit, translator_fn_name)
            translated_circuit, moment_indices = translator_fn(qubit_order)
            self._translated_circuits.append(
                (circuit, translated_circuit, moment_indices)
            )

        return translated_circuit, moment_indices

