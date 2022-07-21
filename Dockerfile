FROM debian

RUN apt-get update
RUN apt-get install -y g++ make wget

COPY ./Makefile /clfsim/Makefile
COPY ./apps/ /clfsim/apps/
COPY ./circuits/ /clfsim/circuits
COPY ./lib/ clfsim/lib/

WORKDIR /clfsim/

RUN make clfsim

ENTRYPOINT ["clfsim/apps/clfsim_base.x"]
