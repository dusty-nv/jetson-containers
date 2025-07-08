#---
# name: ciso8601
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# depends: [build-essential, python]
# test: test.py
# notes: The `ciso8601` wheel that's build is saved in `/opt/wheels`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FORCE_BUILD=off \
    CISO8601_VERSION \
    CISO8601_BRANCH

WORKDIR /

COPY *.diff *.sh /tmp/ciso8601/

RUN /tmp/ciso8601/install.sh || /tmp/ciso8601/build.sh
