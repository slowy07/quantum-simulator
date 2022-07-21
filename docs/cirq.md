# cirq interface

## setting cirq

there are two methods for etting up the clfsim-Cirq interface on your local machine: installing directly with ``pip`` or compiling from the source code.


Prerequisites:

- [CMake](https://cmake.org/): this is used to compile the c++ clfsim librariess. CMake can be installed with ``apt-get install cmake``.
- [Pybind](https://github.com/pybind): this create python wrappers for the c++ libraries, and can be installed with ``pip3 install pybind11``.
- [Cirq](https://cirq.readthedocs.io/en/stable/install.html)

**Compiling clfsimcirq**

1. clone the clfsim repository to local, and navigate to the top level ``clfsim`` directory
```
git clone git@github.com:slowy07/clfsim.git
```
2. compule clfsim usign the top-level Makefile: ``make``. By default, this will use Pybind to generate a static library with file extension ``.so`` in the ``clfsimcirq`` directory.
3. To verify successful compilation, tun the Python test:
```
make run-py test
```

## interface design and operations

the purpose of this interface is to provide a performant simulator for quantum circuit defined in Cirq.


**Classes**

the interface includes CLFSimSimulator and CLFSimhSimulator which communicate through a pybind11 interface with clfsim. The simulator accepts ``cirq.Circuit`` object, which it wraps as ``CLFSimCircuit`` to enforce architectural constraints (such as decomposing to clfsim-supported gate sets).


**usage procedure**

begin by defining a Cirq circuit which to simulate

```python
my_circuit = cirq.Circuit()
```
This circuit can then be simulated using either ``CLFSimSimulator`` or ``CLFSimhSimulator``, depending on the desired output.

