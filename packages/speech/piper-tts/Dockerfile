#---
# name: piper-tts
# group: audio
# depends: [onnxruntime]
# test: [test_piper_phonemize.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PIPER_VERSION=master \
    PIPER_PHONEMIZE_VERSION=master \
    ESPEAK_NG_VERSION=master \
    ONNXRUNTIME_DIR=/usr/local

# make PIPER_CACHE a default mounted location available for downloading the models
# this still needs to be explicitly passed to piper.get_voices(), ensure_path_exists(), ect.
ENV PIPER_CACHE=/data/models/piper \
    PIPER_PHONEMIZE_DIR=/usr/local \
    ESPEAK_NG_DATA_DIR="/opt/espeak-ng/espeak-ng-data"

#ENV LD_LIBRARY_PATH=/usr/local/:$LD_LIBRARY_PATH

WORKDIR /opt

# install espeak-ng prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf automake libtool pkg-config \
        libsonic-dev ronn kramdown \
        libpcaudio-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Even though piper-phonemize will download espeak-ng version
# automatically, we cannot use it because it will not use
# the espeak_TextToPhonemesWithTerminator:
# - https://github.com/rhasspy/piper-phonemize/issues/30
# - https://github.com/espeak-ng/espeak-ng/pull/2127
RUN git clone --branch ${ESPEAK_NG_VERSION} --depth 1 https://github.com/espeak-ng/espeak-ng /opt/espeak-ng && \
    cd /opt/espeak-ng && \
    git fetch origin pull/2127/head:pr-branch && \
    git checkout pr-branch && \
    git status && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    ls -l $ESPEAK_NG_DATA_DIR

# Even though piper will download a piper-phonemize version
# automatically, we cannot use it because it will not use
# our GPU accelerated ONNXRUNTIME. We need to manually build
# piper-phonemize to pass the appropriate flags and have it
# use our ORT (which is already in /usr/local/).
RUN git clone --branch ${PIPER_PHONEMIZE_VERSION} --depth 1 https://github.com/rhasspy/piper-phonemize /opt/piper-phonemize && \
    cd /opt/piper-phonemize && \
    cmake -B build \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DESPEAK_NG_DIR=${ESPEAK_NG_DATA_DIR} \
        -DCMAKE_CXX_FLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" && \
    cmake --build build --config Release --parallel && \
    cmake --install build && \
    CPPFLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" \
    pip3 install --no-cache-dir --verbose . && \
    ln -s ${ESPEAK_NG_DATA_DIR} /usr/share/espeak-ng-data

# Now we can build piper. In order to avoid it from downloading
# automatically piper-phonemize, we specify an installation dir.
RUN git clone --branch ${PIPER_VERSION} --depth 1 https://github.com/rhasspy/piper /opt/piper && \
    cd /opt/piper && \
    cmake -B build -DPIPER_PHONEMIZE_DIR=${PIPER_PHONEMIZE_DIR} -DCMAKE_CXX_FLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" && \
    cmake --build build --config Release --parallel && \
    cmake --install build && \
    cd src/python_run && \
    pip3 install --ignore-installed --no-cache-dir blinker && \
    pip3 install -r requirements_http.txt && \
    pip3 install --no-cache-dir --verbose --no-deps .[gpu,http]

WORKDIR /