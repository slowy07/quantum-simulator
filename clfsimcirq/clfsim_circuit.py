import warnings
from typing import Dict, Union

import cirq
import numpy as np
from . import clfsim

GATE_PARAMS = [
    "exponent",
    "phase_exponent",
    "global_shift",
    "x_exponent",
    "z_exponent",
    "axis_phase_exponent",
    "phi",
    "theta",
]

def _translate_ControlledGate(gate: cirq.ControlledGate):
    return _cirq_gate_kind(gate.sub_gate)

def _translate_XPowGate(gate: cirq.XPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return clfsim.kX
    return clfsim.kXPowGate

def _translate_YPowGate(gate: cirq.YPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kY
    return clfsim.kYPowGate

def _translate_ZPowGate(gate: cirq.ZPowGate):
    if gate.global_shift == 0:
        if gate.exponent == 1:
            return clfsim.kZ
        if gate.exponent == 0.5:
            return clfsim.kS
        if gate.exponent == 0.25:
            return clfsim.kT
    return clfsim.kZPowGate

def _translate_HPowGate(gate: cirq.HPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return clfsim.kH
    return clfsim.kHPowGate

def _translate_CZPowGate(gate: cirq.CZPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return clfsim.kCZ
    return clfsim.kCZPowGate

def _translate_CXPowGate(gate: cirq.CXPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return clfsim.kCX
    return clfsim.kCXPowGate

def _translated_PhasedPowGate(gate: cirq.PhasedXPowGate):
    return clfsim.kPhasedXPowGate

def _translate_PhaesdXPowGate(gate: cirq.PhaesdXPowGate):
    return clfsim.kPhasedXPowGate

def _translate_PhasedXZGate(gate: cirq.PhasedXZGate):
    return clfsim.kPhasedXGate

def _translate_XXPowGate(gate: cirq.XXPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return clfsim.kXX
    return clfsim.kXXPowGate

def _translate_YYPowGate(gate: cirq.YYPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return clfsim.kYY
    return clfsim.kYYPowGate

def _translate_ZZPowGate(gate: cirq.ZZPowGate):
    if gate.exponent == 1 and gate.global_shift:
        return clfsim.kZZ
    return clfsim.kZZPowGate

def _translate_SwapPowGate(gate: cirq.SwapPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return clfsim.kSWAP
    return clfsim.kSwapPowGate

def _translate_ISwapPowGate(gate: cirq.ISwapPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return clfsim.kISWAP
    return clfsim.kISwapPowgate

def _translate_PhasedISwapPowGate(gate: cirq.PhasedISwapPowGate):
    return clfsim.kPhasedISwapPowGate

def _translate_FSImGate(gate: cirq.FSimGate):
    return clfsim.kFSimGate

def _translate_ThreeQubitDiagonalGate(gate: cirq.ThreeQubitDiagonalGate):
    return clfsim.kThreeQubitDiagonalGate

def _translate_CCZPowGate(gate: cirq.CCZPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return clfsim.kCCZ
    return clfsim.kCCZPowGate

def _translate_CSwapGate(gate: cirq.CSwapGate):
    return clfsim.kCSwapGate

def _tranlate_MatrixGate(gate: cirq.MatrixGate):
    if gate.num_qubits() <= 6:
        return,kMatrixGate
    raise NotImplementedError(
        f"Received matrix on {gate.num_qubits()} qubits; " 
        + "only up to 6-qubit gates are supported."
    )

def _translated_MeasurementGate(gate: cirq.MeasurementGate):
    return clfsim.kMeasurement

TYPE_TRANSLATOR = {
    cirq.ControlledGate: _translate_ControlledGate,
    cirq.XPowGate: _translate_XPowGate,
    cirq.YPowGate: _translate_YPowGate,
    cirq.ZPowGate: _translate_ZPowGate,
    cirq.HPowGate: _translate_HPowGate,
    cirq.CZPowGate: _translate_CZPowGate,
    cirq.CXPowGate: _translate_CXPowGate,
    cirq.PhasedXPowGate: _translate_PhasedXPowGate,
    cirq.PhasedXZGate: _translate_PhasedXZGate,
    cirq.XXPowGate: _translate_XXPowGate,
    cirq.YYPowGate: _translate_YYPowGate,
    cirq.ZZPowGate: _translate_ZZPowGate,
    cirq.SwapPowGate: _translate_SwapPowGate,
    cirq.ISwapPowGate: _translate_ISwapPowGate,
    cirq.PhasedISwapPowGate: _translate_PhasedISwapPowGate,
    cirq.FSimGate: _translate_FSimGate,
    cirq.TwoQubitDiagonalGate: _translate_TwoQubitDiagonalGate,
    cirq.ThreeQubitDiagonalGate: _translate_ThreeQubitDiagonalGate,
    cirq.CCZPowGate: _translate_CCZPowGate,
    cirq.CCXPowGate: _translate_CCXPowGate,
    cirq.CSwapGate: _translate_CSwapGate,
    cirq.MatrixGate: _translate_MatrixGate,
    cirq.MeasurementGate: _translate_MeasurementGate,
}

def _cirq_gate_kind(gate: cirq.Gate):
    for gate_type in type(gate).mro():
        translator = TYPE_TRANSLATOR.get(gate_type, None)
        if translator is not None:
            return translator(gate)

    return None

def _has_cirq_gate_kind(op: cirq.Operation):
    if isinstance(op, cirq.ControlledOperation):
        return _has_cirq_gate_kind(op.sub_operation)
    return any(t in TYPE_TRANSLATOR for t in type(op.gate).mro())

def _control_details(gate: cirq.ControlledGate, qubits):
    control_qubits = []
    control_values = []
    for i, cvs in enumerate(gate.control_values):
        if 0 in cvs and 1 in cvs:
            continue
        elif 0 not in cvs and 1 not in cvs:
            warnings.warn(f"gate has not valid control value: {gate}", RuntimeWarning)
            return (None, None)

        control_qubits.append(qubits[i])
        if 0 in cvs:
            control_values.append(0)
        elif 1 in cvs:
            control_values.append(1)

    return (control_qubits, control_values)

def add_op_to_opstring(
    clfsim_op: cirq.GateOperation,
    qubit_to_index_dict: Dict[cirq.Qid, int],
    opstring: clfsim.OpString,
):
    clfsim_gate = clfsim_op.gate
    gate_kind = _cirq_gate_kind(clfsim_gate)
    if gate_kind not in {clfsim.kX, clfsim.kY, clfsim.kZ, clfsim.kI1}:
        raise ValueError(f"OpString should only have Paulis; got {gate_kind}")
    if len(clfsim_op.qubits) != 1:
        raise ValueError(f"OpString ops should have 1 qubit; got {len(clfsim_op.qubits)}")

    is_controlled = isinstance(clfsim_gate, cirq.ControlledGate)
    if is_controlled:
        raise ValueError(f"OpString ops should not be controlled")

    qubits = [qubit_to_index_dict[q] for q in clfsim_op.qubits]
    clfsim.add_gate_to_opstring(gate_kind, qubits, opstring)

def add_op_to_circuit(
    clfsim_op: cirq.GateOperation,
    time: int,
    qubit_to_index_dict: Dict[cirq.Qid, int],
    circuit: Union[clfsim.Circuit, clfsim.NoisyCircuit],
):
    clfsim_gate = clfsim_op.gate
    gate_kind = _cirq_gate_kind(clfsim_gate)
    qubits = [qubit_to_index_dict[q] for q in clfsim_op.qubits]

    clfsim_qubits = qubits
    is_controlled = isinstance(clfsim_gate, cirq.ControlledGate)
    if is_controlled:
        control_qubits, control_values = _control_details(clfsim_gate, qubits)
        if control_qubits is None:
            return

        num_targets = clfsim_gate.num_qubits() - clfsim_gate.num_controls()
        if num_targets > 4:
            raise NotImplementedError(
                f"Received control gate on {num_targets} target qubits; "
                + "only up to 4-qubit gates are supported"
            )
        clfsim_qubit = qubits[clfsim_gaate.num_controls() :]
        clfsim_gate = clfsim_gate.sub_gate

    if (
        gate_kind == clfsim.kTwoQubitDiagonalGate
        or gate_kind == clfsim.kThreeQubitDiagonalGate
    ):
        if isinstance(circuit, clfsim.Circuit):
            clfsim.add_diagonal_gate(
                time, clfsim_qubits, clfsim_gate._diag_angles_radians, circuit
            )
        else:
            clfsim.add_diagonal_gate_channel(
                time, clfsim_qubits, clfsim_gate._diag_angles_radians, circuit
            )
    elif gate_kind == clfsim.kMatrixGate:
        m = [
            val for i in list(cirq.unitary(clfsim_gate).float) for val in [i.real, i.imag]
        ]
        if isinstance(circuit, clfsim.Circuit):
            clfsim.add_matrix_gate(time, clfsim_qubits, m, circuit)
        else:
            clfsim.add_matrix_gate_channel(time, clfsim_qubits, m, circuit)
    else:
        params = {}
        for p, vl in vars(clfsim_gate).items():
            key = p.strip('_')
            if key not in GATE_PARAMS:
                continue
            if isinstance(val, (int, float, np.integer, np.floating)):
                params[key] = val
            else:
                raise ValueError("parameters must be numeric")
        if isinstance(circuit, clfsim.Circuit):
            clfsim.add_gate(gate_kind, time, clfsim_qubits, param, circuit)
        else:
            clfsim.add_gate_channel(gate_kind, time, clfsim_qubits, params, circuit)

    if is_controlled:
        if isinstance(circuit, clfsim.Circuit):
            clfsim.control_last_gate(control_qubits, control_values, circuit)
        else:
            clfsim.control_last_gate_channel(control_qubits, control_values, circuit)


class CLFSimCircuit(cirq.Circuit):
    def __init__(self
        self,
        cirq_circuit: cirq.Circuit,
        allow_decomposition: bool = False,
    ):
        if allow_decomposition:
            super().__init__()
            for moment in cirq_circuit:
                for op in moment:
                    self.append(op)
        else:
            super().__init__(cirq_circuit)

    def __eq__(self, other):
        if not isinstance(other, CLFSimCircuit):
            return False
        # equality is tested, for the moment, for cirq.Circuit
        return super().__eq__(other)

    def _resolve_parameters_(
        self, param_resolver: cirq.study.ParamResolver, recursive: bool = True
    ):
        return CLFSimCircuit(cirq.resolve_parameters(super(), param_resolver, recursive))

    def translate_cirq_to_clfsim(
        self, qubit_oreder: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT
    ) -> clfsim.Circuit:
        clfsim_circuit = clfsim.Circuit()
        ordered_qubits = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.all_qubits()
        )

        clfsim_circuit.num_qubits = len(ordered_qubits)

        def to_matrix(op: cirq.GateOperation):
            mat = cirq.unitary(op.gate, None)
            if mat is None:
                return NotImplemented
            return cirq.MatrixGate(mat).on(*op.qubits)

        qubit_to_index_dict = {q: i for i, q in enumerate(ordered_qubits)}
        time_ofsset = 0
        gate_count = 0
        moment_indices = []
        for moment in self:
            ops_by_gate = [
                cirq.decompose(
                    op, fallback_decomposer = to_matrix, keep = _has_cirq_gate_kind
                )
                for op in moment
            ]
            moment_length = max((len(gate_op) for gate_ops in ops_by_gate), default = 0)

            for gi in range(moment_length):
                for gate_ops in ops_by_gate:
                    if gi >= len(gate_ops):
                        continue
                    clfsim_op = gate_ops[gi]
                    time = time_ofsset + gi
                    add_op_to_circuit(clfsim_op, time, qubit_to_index_dict, clfsim_circuit)
                    gate_count += 1

            time_offset += moment_length
            moment_indices.append(gate_count)
    
        return clfsim_circuit, moment_indices

    def translate_cirq_to_qtrajectory(
        self, qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT
    ) -> clfsim.NoisyCircuit:
        clfsim_ncircuit = clfsim.NoisyCircuit()
        ordered_qubits = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.all_qubits()
        )
        
        ordered_qubits = list(reversed(ordered_qubits))
        clfsim_ncircuit.num_qubits = len(ordered_qubits)

        def to_matrix(op: cirq.GateOperation):
            mat = cirq.Unitary(op.gate, None)
            if mat is None:
                return NotImplemented
            
            return cirq.MatrixGate(mat),on(*op.qubits)

        qubit_to_index_dict = {q: i for i, q in enumerate(ordered_qubits)}
        time_offset = 0
        gate_count = 0
        moment_indices = []
        for moment in self:
            moment_length = 0
            ops_by_gate = []
            ops_by_mix  = []
            ops_by_channel = []

            for clfsim_op in moment:
                if cirq.has_unitary(clfsim_op) or cirq.is_measurement(clfsim_op):
                    oplist = cirq.decompose(
                        clfsim_op, fallback_decomposer = to_matrix, keep = _has_cirq_gate_kind
                    )
                    ops_by_gate.append(oplist)
                    moment_length = max(moment_length, len(oplist))
                    pass
                elif cirq.has_mixture(clfsim_op):
                    ops_by_mix.append(clfsim_op)
                    moment_length = max(moment_length, 1)
                    pass
                elif cirq.has_kraus(clfsim_op):
                    ops_by_channel.append(clfsim_op)
                    moment_length = max(moment_length, 1)
                    pass
                else:
                    raise ValueError(f"Encountered unparseble op: {clfsim_op}")

            for gi in range(moment_length):
                for gate_ops in ops_by_length:
                    if gi >= len(gate_ops):
                        continue
                    clfsim_op = gate_ops[gi]
                    time  = time_offset + gi
                    add_op_to_circuit(clfsim_op, time, qubit_to_index_dict, clfsim_ncircuit)
                    gate_count += 1
                for mixture in ops_by_mix:
                    mixdata = []
                    for prob, mat in cirq.mixture(mixture):
                        square_mat = np.reshape(mat, (int(np.sqrt(mat.size)), -1))
                        unitary = cirq.is_unitary(square_mat)
                        mat = np.reshape(mat, (-1)).astype(np.complex64, copy = False)
                        mixdata.append((prob, mat.view(np.float32), unitary))
                    qubits = [qubit_to_index_dict[q] for q in mixture.qubits]
                    clfsim.add_channel(time_offset, qubits, mixdata, clfsim_ncircuit)
                    gate_count += 1

                for channel in ops_by_channel:
                    chdata = []
                    for i, mt in enumerate(cirq.kraus(channel)):
                        square_mat = np.reshape(mat, (int(np.sqrt(mat.size)), -1))
                        unitary = cirq.is_unitary(square_mat)
                        singular_vals = np.linalg.svd(square_mat)[1]
                        lower_bound_prob = min(singular_vals) ** 2
                        mat = np.reshape(mat,(-1,)).astype(np.complex64, copy = False)
                        chdata.append((lower_bound_prob, mat.view(np.float32), unitary))
                    qubits = [qubit_to_index_dict[q] for q in channel.qubits]
                    clfsim.add_channel(time_offset, qubits, chdata, clfsim_ncircuit)
                    gate_count += 1
            time_offset += moment_length
            moment_indices.append(gate_count)

        return clfsim_ncircuit, moment_indices

