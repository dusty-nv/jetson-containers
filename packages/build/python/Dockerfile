#---
# name: python
# group: build
# depends: [build-essential, pip_cache]
# notes: installs core `python3` packages and `pip`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION \
    TMP=/tmp/python

ENV PYTHON_VERSION=${PYTHON_VERSION} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=true \
    PIP_CACHE_PURGE=true \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_WHEEL_DIR=/opt/wheels \
    PIP_VERBOSE=1 \
    TWINE_NON_INTERACTIVE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH

#PYTHONPATH=/opt/venv/lib/python${PYTHON_VERSION_ARG}/site-packages:/usr/lib/python3/dist-packages:$PYTHONPATH \

COPY install.sh $TMP/
RUN $TMP/install.sh
