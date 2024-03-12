#---
# name: torchaudio
# group: pytorch
# config: config.py
# depends: [pytorch, cmake]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHAUDIO_VERSION

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  git \
		  pkg-config \
		  libffi-dev \
		  libsndfile1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    pip3 install --no-cache-dir --verbose pysoundfile && \
    \
    git clone --branch ${TORCHAUDIO_VERSION} --recursive --depth=1 https://github.com/pytorch/audio /opt/torchaudio && \
    cd /opt/torchaudio && \
    git checkout ${TORCHAUDIO_VERSION} && \
    \
    # note:  see https://github.com/pytorch/audio/issues/2295 for the reason for the sed commands
    sed -i 's#  URL https://zlib.net/zlib-1.2.11.tar.gz#  URL https://github.com/madler/zlib/archive/refs/tags/v1.2.12.tar.gz#g' third_party/zlib/CMakeLists.txt || echo "failed to patch torchaudio/third_party/zlib/CMakeLists.txt" && \
    sed -i 's#  URL_HASH SHA256=c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1#  URL_HASH SHA256=d8688496ea40fb61787500e863cc63c9afcbc524468cedeb478068924eb54932#g' third_party/zlib/CMakeLists.txt || echo "failed to patch torchaudio/third_party/zlib/CMakeLists.txt" && \
    \
    BUILD_SOX=1 python3 setup.py bdist_wheel && \
    cp dist/torchaudio*.whl /opt && \
    cd ../ && \
    rm -rf torchaudio && \
    \
    pip3 install --no-cache-dir --verbose /opt/torchaudio*.whl && \
    python3 -c 'import torchaudio; print(torchaudio.__version__);'