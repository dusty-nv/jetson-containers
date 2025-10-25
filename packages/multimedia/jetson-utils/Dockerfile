#---
# name: jetson-utils
# group: multimedia
# depends: [cuda, cudastack:standard, python, cmake, gstreamer]
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG JETSON_UTILS_REPO=dusty-nv/jetson-utils \
    JETSON_UTILS_BRANCH=master \
    JETSON_UTILS_CMAKE=on \
    JETSON_UTILS_PIP=on

RUN if [ "$JETSON_UTILS_CMAKE" = "on" ]; then \
        apt-get update ; \
        apt-get install -y --no-install-recommends \
		    libglew-dev \
		    glew-utils \
		    libsoup2.4-dev \
		    libjson-glib-dev \
        libgstrtspserver-1.0-0 \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-nice \
        libnss-mdns \
		    libgstrtspserver-1.0-dev \
		    avahi-utils && \
        rm -rf /var/lib/apt/lists/* && \
        apt-get clean ; \
    fi

ADD https://api.github.com/repos/${JETSON_UTILS_REPO}/git/refs/heads/${JETSON_UTILS_BRANCH} /tmp/jetson_utils_version.json

RUN git clone --branch=${JETSON_UTILS_BRANCH} --depth=1 https://github.com/${JETSON_UTILS_REPO} /opt/jetson-utils

RUN if [ "$JETSON_UTILS_CMAKE" = "on" ]; then \
        cd /opt/jetson-utils ; \
        mkdir build ; \
        cd build ; \
        cmake ../ ; \
        make -j$(nproc) ; \
        make install -j$(nproc) ; \
    fi

RUN if [ "$JETSON_UTILS_PIP" = "on" ]; then \
        cd /opt/jetson-utils ; \
        /opt/venv/bin/python -m pip install --no-cache-dir -e . ; \
    fi

RUN LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/" /opt/venv/bin/python -c 'import jetson_utils'
