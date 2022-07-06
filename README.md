# quantum-simulator

Quantum circuit simulator clfsim and clfsimh. these simulator were used for cross entropy benchmarking

## clfsim

clfim is a Schrödinger full state-vector simulator It computes all the 2n amplitudes of the state vector, where n is the number of qubits. Essentially, the simulator performs matrix-vector multiplications repeatedly. One matrix vector multiplication coresspond to applying one gate. The total runtime is propotional to g^2^n, wehere _g_ is the number of 2-qubit gates. To speed up the fusion, using gate fusion, single precision arithmetic.


## related paper
- M. Smelyanskiy, N. P. Sawaya, A. Aspuru-Guzik, "qHiPSTER: The Quantum High Performance Software Testing Environment", arXiv:1601.07195 (2016).

- T. Häner, D. S. Steiger, "0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit", arXiv:1704.01127 (2017).