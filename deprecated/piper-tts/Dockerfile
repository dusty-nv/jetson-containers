#---
# name: piper-tts
# group: audio
# depends: [cuda, torchaudio, onnxruntime, espeak]
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
    PIPER_PHONEMIZE_DIR=/usr/local

# Even though piper will download a piper-phonemize version
# automatically, we cannot use it because it will not use
# our GPU accelerated ONNXRUNTIME. We need to manually build
# piper-phonemize to pass the appropriate flags and have it
# use our ORT (which is already in /usr/local/).
RUN git clone --branch ${PIPER_PHONEMIZE_VERSION} --depth 1 https://github.com/rhasspy/piper-phonemize /opt/piper-phonemize && \
    cd /opt/piper-phonemize && \
    sed -i 's|__version__ = "1.2.0"|__version__ = "1.1.99"|' setup.py && \
    cmake -B build \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DESPEAK_NG_DIR=${ESPEAK_NG_DATA_DIR} \
        -DCMAKE_CXX_FLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" && \
    cmake --build build --config Release --parallel && \
    cmake --install build && \
    pip install pybind11 && \
    CPPFLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" \
    uv pip install . --no-deps --no-build-isolation && \
    ln -s ${ESPEAK_NG_DATA_DIR} /usr/share/espeak-ng-data

RUN echo "✅ piper-phonemize version:" && pip show piper-phonemize | grep -i version

# Now we can build piper. In order to avoid it from downloading
# automatically piper-phonemize, we specify an installation dir.
RUN git clone --branch ${PIPER_VERSION} --depth 1 https://github.com/rhasspy/piper /opt/piper && \
    cd /opt/piper && \
    cmake -B build -DPIPER_PHONEMIZE_DIR=${PIPER_PHONEMIZE_DIR} -DCMAKE_CXX_FLAGS="-I${ONNXRUNTIME_DIR}/include/onnxruntime/" && \
    cmake --build build --config Release --parallel && \
    cmake --install build && \
    cd src/python_run && \
    uv pip install --reinstall blinker && \
    uv pip install -r requirements_http.txt && \
    uv pip install --no-deps .[gpu,http]
