# clfsim and clfsimh

clfsim and clfsimh are a collection of C++ libraries for quantum circuit simulation. These libraries provide powerful, low-cost tools for researchers to test quantum algorithms before running on quantum hardware.

clfsim makes of AVX/FMA vector operations, OpenMP multithreading, and gate fusion to accelerate simulations. This performance is best demonstrated by the use of clfsim in cross-entropy benchmarks [here](https://www.nature.com/articles/s41586-019-1666-5)

Integration with [Cirq](https://quantumai.google/cirq) makes getting started with clfsim easy.

## Design

this repository includes two top-level libraries for simulation:

- **clfsim** is a Schrödinger state-vector simulator designed to run on a single machine. It produces the full state vector as output wich, for instance, allow users to sample repeatedly from a single execution.
- **clfsimh** is a hybrid Schrödinger-Feynman simulator build for parallel execution on a clutser of machines. It produces amplitudess for user-specified output bitstrings.

These libraries can be invoked either directly or through the clfsim-Cirq interface to perform the following operations:

- Determine the final state vector of a circuit (clfsim only)
- Sample result from a circuit. Multiple samples can be generated with minimal addicitional cost for circuit with no intermediate measurements (clfsim only).
- Calculate amplitudes for user-specified result bitstrings. with clfsimh, this trivially parallelizable across several machines.

Circuit of up to 30 qubits can be simulated in clfsim with ~16GB of RAM; each additional qubit doubles RAM requirement. In contrast, carefull use of clfsimh can support 50 qubit or more.
