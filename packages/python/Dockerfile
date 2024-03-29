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
    PYTHONIOENCODING=utf-8 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_CACHE_PURGE=true \
    PIP_ROOT_USER_ACTION=ignore \
    TWINE_NON_INTERACTIVE=1

COPY install.sh /tmp/install_python.sh 
RUN /tmp/install_python.sh
