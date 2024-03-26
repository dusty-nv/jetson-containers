#---
# name: build-essential
# group: build
# notes: installs compilers, build tools & configures the default locale
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANGUAGE=en_US:en \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TERM=dumb

RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        locales \
        locales-all \
        tzdata \
    && locale-gen en_US $LANG \
    && update-locale LC_ALL=$LC_ALL LANG=$LANG \
    && locale \
    \
    && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        lsb-release \
        pkg-config \
        gnupg \
        git \
        gdb \
        wget \
        curl \
        nano \
        zip \
        unzip \
        time \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    \
    && gcc --version \
    && g++ --version
