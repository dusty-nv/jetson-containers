#---
# name: devpi
# group: net
# notes: installs local devpi server for hosting pip wheels
#---
ARG BASE_IMAGE="python:3.10"
FROM ${BASE_IMAGE}

RUN pip3 install 'devpi-server<6.13' \
        devpi-client devpi-web \
        devpi-constrained \
        devpi-findlinks \
        devpi-cleaner \
        devpi-slack

COPY model.patch /tmp/model.patch

RUN PYTHON_ROOT=`pip show devpi-server | grep Location: | cut -d' ' -f2` && \
    ls $PYTHON_ROOT/devpi* && \
    cd $PYTHON_ROOT/devpi_server && \
    patch -p1 < /tmp/model.patch && \
    cat model.py | grep 'DEVPI_MIRROR_BLACKLIST' && \
    mkdir /devpi

CMD devpi-server --listen 0.0.0.0:80 \
        --max-request-body-size 4294967296 \
        --serverdir /devpi \
        --indexer-backend null

EXPOSE 80