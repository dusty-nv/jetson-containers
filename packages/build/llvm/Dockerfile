#---
# name: llvm
# alias: clang
# group: build
# depends: [build-essential]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LLVM_VERSION \
    TMP=/tmp/llvm

COPY install.sh $TMP/

RUN cd $TMP && ./install.sh

ENV LLVM_VERSION=${LLVM_VERSION} \
    LLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}

#CC=/usr/lib/llvm-${LLVM_VERSION}/bin/clang
#CXX=/usr/lib/llvm-${LLVM_VERSION}/bin/clang++