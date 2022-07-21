# input circuit file format

**WARNING**: this format only support the ``gate_clfsim`` gate set, and is no longer actively maintained. For other gates, circuit must be defined in code or through the clfsimcirq interface using Cirq.

the first line contains the number of qubits. the rest of the lines specify gates with one gate per line. The format for a gate is

```
time gate_name qubits parameters
```

Here ``time`` refers to when the gates i applied in the circuit. gates with the same time can be applied independently and they may be reordered for performance. Trailing spaces or character are not allowd. A number of sample circuit are provided in ``circuits``.
