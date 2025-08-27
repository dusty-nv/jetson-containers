#---
# name: go
# group: build
# depends: [build-essential, python]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# default Go version can be overridden by GOLANG_VERSION build arg
ARG GOLANG_VERSION="1.22.7"

RUN curl -fsSL https://go.dev/dl/go${GOLANG_VERSION}.linux-arm64.tar.gz -o /tmp/go.tgz \
 && rm -rf /usr/local/go \
 && tar -C /usr/local -xzf /tmp/go.tgz \
 && rm /tmp/go.tgz

ENV PATH="/usr/local/go/bin:${PATH}" \
    GOPATH="/root/go"

RUN go version



