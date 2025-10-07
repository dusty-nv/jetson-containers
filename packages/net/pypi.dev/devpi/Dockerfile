#---
# name: devpi
# group: net
# notes: installs local devpi server for hosting pip wheels
#---
ARG BASE_IMAGE="python:3.12"
FROM ${BASE_IMAGE}

RUN uv pip install 'devpi-server<6.16' \
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

COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]

EXPOSE 80
