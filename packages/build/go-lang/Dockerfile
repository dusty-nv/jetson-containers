#---
# name: go-lang
# group: build
# depends: [build-essential]
# notes: installs go lang
# test: test.sh
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GOBIN=/opt/go/bin

# Install golang
RUN set -ex \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends golang \
	&& apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set Go env variables
ENV GOPATH=/opt/go \
	PATH=$PATH:$GOBIN

# Test golang version
RUN set -ex \
	&& go version \
	&& go env
