FROM debian

RUN apt-get update
RUN apt-get install -y python3-dev python3-pip
RUN apt-get install -y cmake git

COPY ./ /clfsim/
RUN pip3 install /clfsim/

COPY ./clfsimcirq_test/ /test-install/

WORKDIR /test-install/

ENTRYPOINT python3 -m pytest ./
