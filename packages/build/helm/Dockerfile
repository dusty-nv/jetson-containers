#---
# name: helm
# group: build
# depends: [build-essential]
# test: test.sh
#---
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

RUN set -ex \
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 \
    && chmod 700 get_helm.sh \
    && ./get_helm.sh

ENV PATH=/usr/local/bin/helm:$PATH
