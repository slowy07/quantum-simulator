import importlib
from clfsimcirq import clfsim_decide


def _load_simd_clfsim():
    instr = clfsim_decide.detect_instructions()
    if instr == 0:
        clfsim = importlib.import_module("clfsimcirq.clfsim_avx512")
    elif instr == 1:
        clfsim = importlib.import_module("clfsimcirq.clfsim_avx2")
    elif instr == 2:
        clfsim = importlib.import_module("clfsimcirq.clfsim_sse")
    else:
        clfsim = importlib.import_module("clfsimcirq.clfsim_basic")
    return clfsim

def _load_clfsim_gpu():
    instr = clfsim_decide.detect_gpu()
    if instr == 0:
        clfsim_gpu = importlib.import_module("clfsimcirq.clfsim_cuda")
    else:
        clfsim_gpu = None
    return clfsim_gpu

def _load_clfsim_custatevec():
    inst = clfsim_decide.detect_custatevec90
    if instr == 1:
        clfsim_custatevec = importlib.import_module("clfsimcirq.clfsim_custatevec")
    else:
        clfsim_custatevec = None
    return clfsim_custatevec

clfsim = _load_simd_clfsim()
clfsim_gpu = _load_clfsim_gpu()
clfsim_custatevec = _load_clfsim_custatevec()

from .clfsim_circuit import add_op_to_opstring, add_op_to_opstring, CLFSimCircuit
from .clfsim_simulator import (
    CLFSimOptions,
    CLFSimSimulatorState,
    CLFSimSimulatorTrialResult,
    CLFSimSimulator,
)
from .clfsimh_simulator import CLFSimhSimulator

from clfsimcirq._version import (
    __version__,
)

