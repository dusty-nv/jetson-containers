#---
# name: vscode
# group: code
# depends: [build-essential, python]
# notes: installs portable build of VSCode inside container
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VSCODE_URL \
    VSCODE_TAR=/tmp/vscode.tar.gz

ENV VSCODE_ROOT=/opt/vscode \
    VSCODE_DATA=/opt/vscode/data

# https://code.visualstudio.com/docs/editor/portable
RUN wget $WGET_FLAGS $VSCODE_URL -O $VSCODE_TAR && \
    tar -xzvf $VSCODE_TAR -C $VSCODE_ROOT && \
    rm $VSCODE_TAR && \
    mkdir $VSCODE_ROOT/data && \
    ln -s $VSCODE_ROOT/bin/code /usr/local/bin/code
