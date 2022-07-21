EIGEN_PREFIX = "d10b27fe37736d2944630ecd7557cefa95cf87c9"
EIGEN_URL = "https://github.com/libeigen/eigen/-/archive/"

TARGETS = clfsim
TEST = run-cxx-test

CXX=g++
NVCC=nvcc

CXXFLAGS = -O3 -fopenmp
ARCHFLAGS = -march=native
NVCCFLAGS = -O3

CUSTATEVECFLAGS = -i$(QUANTUM_DIR)/include -L${CUQUANTUM_DIR}/lib -lcustatevec -lcublas

PYBIND11 = true

export CXX
export CXXFLAGS
export ARCHFLAGS
export NVCC
export NVCCFLAGS
export CUSTATEVECFLAGS

ifeq ($(PYBIND11), true)
	TARGETS += pybind
	TEST += run-py-test
endif

.PHONY: all
all: $(TARGETS)

.PHONY: clfsim
clfsim:
	$(MAKE) -C apps/ clfsim

.PHONY: clfsim-cuda
clfsim-cuda:
	$(MAKE) -C apps/ clfsim-cuda

.PHONY: clfsim-custatevec
clfsim-custatevec:
	$(MAKE) -C apps/ clfsim-custatevec

.PHONY: pybind
pybind:
	$(MAKE) -C pybind_interface/ pybind

.PHONY: cxx-test
cxx-test: eigen
	$(MAKE) -C test/ cxx-test

.PHONY: cuda-test
cuda-test:
	$(MAKE) -C test/ cuda-test

.PHONY: custatevec-test
custatevec-test:
	$(MAKE) -C tests/ custatevec-test

.PHONY: run-cxx-test
run-cxx-test: cxx-test
	$(MAKE) -C test/ run-cxx-test

.PHONY: run-cuda-test
run-cuda-test: cuda-test
	$(MAKE) -C test/ run-cuda-test

.PHONY: run-custatevec-test
run-custatevec-test: custatevec-test
	$(MAKE) -C test/ run-custatevec-test

PYTEST = $(shell find clfsimcirqq_test/ -name '*_test.py')

.PHONY: run-py-test
run-py-test: pybind
	for exe in $(PYTEST); do if ! python3 -m pytest $$exe; then exit 1; fi; done

.PHONY: run-test
run-test: $(TEST)

eigen:
	$(shell\
		rm -rf eigen;\
		wget $(EIGEN_URL)/$(EIGEN_PREFIX)/eigen-$(EIGEN_PREFIX).tar.gz;\
		tar -xf eigen-$(EIGEN_PREFIX).tar.gz && mv eigen-$(EIGEN_PREFIX) eigen; \
		rm eigen-$(EIGEN_PREFIX).tar.gz;)

.PHONY: clean
clean:
	rm -rf eigen
	$(MAKE) -C apps/ clean
	-$(MAKE) -C tests/ clean
	-$(MAKE) -C pybind_interface/ clean

