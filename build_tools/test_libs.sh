#!/bin/sh

set -e
set -x

which bazel
bazel version

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  bazel build --config=sse apps:all
  bazel build apps:all
fi

set e
EXIT_CODE=0
for TARGET in bitstring_test channels_cirq_test circuit_clfsim_parser_test expect_test \
              fuser_basic_test gates_clfsim_test hybrid_test matrix_test qtrajectory_test \
              run_clfsim_test run_clfsimh_test simulator_basic_test simulator_sse_test statespace_basic_test \
              statespace_sse_test unitary_calculator_basic_test unitary_calculator_sse_test \
              unitaryspace_basic_test unitaryspace_sse_test vectorspace_test; do \
  if ! bazel test --test_output=errors tests:${TARGET}; then
    EXIT_CODE=1
  fi
done
