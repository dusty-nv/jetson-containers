#---
# name: piper-tts
# group: audio
# depends: [cuda, cudnn, tensorrt, cmake, python, numpy, onnx, onnxruntime]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Even though piper will download a piper-phonemize version
# automatically, we cannot use it because it will not use
# our GPU accelerated ONNXRUNTIME. We need to manually build
# piper-phonemize to  pass the appropriate flags and have it
# use our ORT (which is already in /usr/local/).
ARG PIPER_PHONEMIZE_VERSION=master
ARG ONNXRUNTIME_DIR=/usr/local
ENV LD_LIBRARY_PATH=/usr/local/:$LD_LIBRARY_PATH
WORKDIR /
RUN git clone https://github.com/rhasspy/piper-phonemize --depth 1 -b ${PIPER_PHONEMIZE_VERSION} && \
    cd piper-phonemize && \
    cmake -Bbuild -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} -DCMAKE_CXX_FLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" && \
    cmake --build build --config Release && \
    cmake --install build && \
    CPPFLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" pip3 install --no-cache-dir --verbose . && \
    ln -s /usr/local/share/espeak-ng-data /usr/share/espeak-ng-data

# Now we can build piper. In order to avoid it from downloading
# automatically piper-phonemize, we specify an installation dir.
ARG PIPER_VERSION=master
ARG PIPER_PHONEMIZE_DIR=/usr/local
WORKDIR /
RUN git clone https://github.com/rhasspy/piper --depth 1 -b ${PIPER_VERSION} && \
    cd piper && \
    cmake -Bbuild -DPIPER_PHONEMIZE_DIR=${PIPER_PHONEMIZE_DIR} -DCMAKE_CXX_FLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" && \
    cmake --build build --config Release && \
    cmake --install build && \
    cd src/python_run && pip3 install -r requirements_http.txt && \
    pip3 install --no-cache-dir --verbose --no-deps .[gpu,http]
