#---
# name: cangaroo
# group: hardware
# depends: [canbus, cmake]
# test: test.sh
# notes: CAN bus analyzer (github.com/normaldotcom/cangaroo)
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        qt5-qmake \
        qtbase5-dev \
        libnl-3-dev \
        libnl-route-3-dev \
        libqt5serialport5 \
        libqt5serialport5-dev \
        libqt5charts5 \
        libqt5charts5-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN git clone --depth=1 https://github.com/normaldotcom/cangaroo /opt/cangaroo && \
    cd /opt/cangaroo && \
    qmake -qt=qt5 && \
    make -j$(nproc) && \
    make install && \
    ln -s /opt/cangaroo/bin/cangaroo /usr/local/bin/cangaroo