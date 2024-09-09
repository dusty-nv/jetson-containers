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
    ONNXRUNTIME_DIR=/usr/local

# make PIPER_CACHE a default mounted location available for downloading the models
# this still needs to be explicitly passed to piper.get_voices(), ensure_path_exists(), ect.
ENV PIPER_CACHE=/data/models/piper \
    PIPER_PHONEMIZE_DIR=/usr/local \
    ESPEAK_NG_DATA_DIR=/usr/local/share/espeak-ng-data

#ENV LD_LIBRARY_PATH=/usr/local/:$LD_LIBRARY_PATH

WORKDIR /opt

# Even though piper will download a piper-phonemize version
# automatically, we cannot use it because it will not use
# our GPU accelerated ONNXRUNTIME. We need to manually build
# piper-phonemize to  pass the appropriate flags and have it
# use our ORT (which is already in /usr/local/).
RUN git clone --branch ${PIPER_PHONEMIZE_VERSION} --depth 1 https://github.com/rhasspy/piper-phonemize && \
    cd piper-phonemize && \
    cmake -B build -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} -DCMAKE_CXX_FLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" && \
    cmake --build build --config Release --parallel && \
    cmake --install build && \
    CPPFLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" \
    pip3 install --no-cache-dir --verbose . && \
    ln -s ${ESPEAK_NG_DATA_DIR} /usr/share/espeak-ng-data
 
# Now we can build piper. In order to avoid it from downloading
# automatically piper-phonemize, we specify an installation dir.
RUN git clone --branch ${PIPER_VERSION} --depth 1 https://github.com/rhasspy/piper && \
    cd piper && \
    cmake -B build -DPIPER_PHONEMIZE_DIR=${PIPER_PHONEMIZE_DIR} -DCMAKE_CXX_FLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" && \
    cmake --build build --config Release --parallel && \
    cmake --install build && \
    cd src/python_run && \
    pip3 install --ignore-installed --no-cache-dir blinker && \
    pip3 install -r requirements_http.txt && \
    pip3 install --no-cache-dir --verbose --no-deps .[gpu,http]

WORKDIR /
