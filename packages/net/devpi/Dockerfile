#---
# name: devpi
# group: net
# notes: installs local devpi server for hosting pip wheels
#---
ARG BASE_IMAGE="python:3.14"
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates patch procps \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install devpi-server \
        devpi-client \
        devpi-web \
        devpi-constrained \
        devpi-findlinks \
        devpi-cleaner \
        devpi-lockdown \
        devpi-tokens \
        devpi-slack

COPY model.patch /tmp/model.patch

RUN PYTHON_ROOT=`pip show devpi-server | grep Location: | cut -d' ' -f2` && \
    ls $PYTHON_ROOT/devpi* && \
    cd $PYTHON_ROOT/devpi_server && \
    patch -p1 < /tmp/model.patch && \
    cat model.py | grep 'DEVPI_MIRROR_BLACKLIST' && \
    mkdir /devpi

COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]

EXPOSE 3141    