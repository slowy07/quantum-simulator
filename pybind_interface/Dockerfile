FROM clfsim
RUN apt-get install -y python3-dev python3-pybind11 python3-pytest python3-pip

RUN pip3 install --prefer-binary cirq-core --force

COPY ./pybind_interface/ /clfsim/pybind_interface/
COPY ./clfsimcirq /clfsim/clfsimcirq/
COPY ./clfsimcirq_test /clfsim/clfsimcirq_test/

WORKDIR /clfsim/

RUN make -C /clfsim pybind
ENTRYPOINT make -C /clfsim/ run-py-test

