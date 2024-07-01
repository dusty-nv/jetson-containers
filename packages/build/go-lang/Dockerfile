#---
# name: go-lang
# group: build
# config: config.py
# depends: [build-essential]
# notes: installs go lang
# test: test.sh
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GO_VERSION \
	GO_INSTALL_DIR=/usr/local

# Install golang
RUN wget --quiet --show-progress --progress=bar:force:noscroll \
		https://golang.org/dl/go${GO_VERSION}.linux-arm64.tar.gz \
		-O /tmp/go${GO_VERSION}.linux-arm64.tar.gz \
	&& rm -rf ${GO_INSTALL_DIR}/go \
	&& tar -C ${GO_INSTALL_DIR} -xzf /tmp/go${GO_VERSION}.linux-arm64.tar.gz \
	&& rm /tmp/go${GO_VERSION}.linux-arm64.tar.gz

# set Go env variables
ENV PATH="$PATH:${GO_INSTALL_DIR}/go/bin" \
	GOOS=linux \
	GOARCH=arm64

# Test golang version
RUN set -ex \
	&& go version \
	&& go env
