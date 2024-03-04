#---
# name: nodejs
# group: build
# depends: [build-essential]
# test: test.sh
# notes: installs `nodejs`, `npm`
#---
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG NODE_MAJOR=20

RUN set -ex \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PATH=/usr/bin/node:$PATH

RUN set -ex \
    && node --version \
    && npm --version
