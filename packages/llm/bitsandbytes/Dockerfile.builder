#---
# name: bitsandbytes:builder
# group: llm
# requires: '>=35'
# config: config.py
# depends: [transformers]
# test: test.py
# notes: fork of https://github.com/TimDettmers/bitsandbytes for Jetson
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# upstream version is https://github.com/TimDettmers/bitsandbytes (the fork below includes some patches for Jetson) 
ARG BITSANDBYTES_REPO \
    BITSANDBYTES_BRANCH \
    CUDA_INSTALLED_VERSION \
    CUDA_MAKE_LIB

ADD https://api.github.com/repos/${BITSANDBYTES_REPO}/git/refs/heads/${BITSANDBYTES_BRANCH} /tmp/bitsandbytes_version.json

RUN set -ex \
    && echo "CUDA_INSTALLED_VERSION: $CUDA_INSTALLED_VERSION" \
    && echo "CUDA_MAKE_LIB: $CUDA_MAKE_LIB" \
    && pip3 uninstall -y bitsandbytes \
    && git clone --depth=1 "https://github.com/$BITSANDBYTES_REPO" /opt/bitsandbytes \
    && CUDA_VERSION=$CUDA_INSTALLED_VERSION make -C /opt/bitsandbytes -j$(nproc) "${CUDA_MAKE_LIB}" \
    && CUDA_VERSION=$CUDA_INSTALLED_VERSION make -C /opt/bitsandbytes -j$(nproc) "${CUDA_MAKE_LIB}_nomatmul" \
    && cd /opt/bitsandbytes \
    && python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel --dist-dir /opt \
#    && rm -rf /opt/bitsandbytes \
    && ls -l /opt/ \
    && pip3 install --no-cache-dir --verbose \
        scipy \
        /opt/bitsandbytes*.whl \
    \
    && pip3 show bitsandbytes \
    && python3 -c 'import bitsandbytes'
