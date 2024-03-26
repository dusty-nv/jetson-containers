#---
# name: python
# alias: python3
# group: build
# depends: build-essential
# notes: installs core `python3` packages and `pip`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEADSNAKES_PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=true \
    PIP_CACHE_PURGE=true \
    PIP_ROOT_USER_ACTION=ignore

RUN set -ex \
    && apt-get update \
    && if [ -n "${DEADSNAKES_PYTHON_VERSION}" ]; then \
        echo "DEADSNAKES_PYTHON_VERSION: ${DEADSNAKES_PYTHON_VERSION}" \
        && add-apt-repository ppa:deadsnakes/ppa \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
            python${DEADSNAKES_PYTHON_VERSION} \
            python${DEADSNAKES_PYTHON_VERSION}-dev \
        && curl -sS https://bootstrap.pypa.io/get-pip.py | python${DEADSNAKES_PYTHON_VERSION} \
        && ln -s /usr/bin/python${DEADSNAKES_PYTHON_VERSION} /usr/local/bin/python3 \
        && ln -s /usr/bin/pip${DEADSNAKES_PYTHON_VERSION} /usr/local/bin/pip3 \
        && python3 --version \
        && pip3 install --no-cache-dir --verbose --no-binary :all: psutil \
    ; else \
        apt-get install -y --no-install-recommends \
            python3-dev \
            python3-pip \
        && ln -s /usr/bin/python3 /usr/local/bin/python \
        && ln -s /usr/bin/pip3 /usr/local/bin/pip \
        && pip3 install --no-cache-dir --verbose psutil \
    ; fi \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && which pip3 \
    && pip3 --version \
    && python3 -m pip install --upgrade pip \
    # this was causing issues downstream (e.g. Python2.7 still around in Ubuntu 18.04, \
    # and in cmake python enumeration where some packages expect that 'python' is 2.7) \
    #RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \  \
    #    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && which python || python --version || pip --version || which python3 || python3 --version || pip3 --version \
    && pip3 install --upgrade --no-cache-dir \
        setuptools \
        packaging \
        'Cython<3' \
        wheel \
