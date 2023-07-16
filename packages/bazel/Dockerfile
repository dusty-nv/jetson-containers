#
# name: bazel
# category: base
# depends: [build-essential]
# test: test.sh
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


# https://github.com/bazelbuild/bazelisk
RUN BAZELISK_RELEASE=$(wget -qO- https://api.github.com/repos/bazelbuild/bazelisk/releases/latest | grep -Po '"tag_name": "\K.*?(?=")') && \
    BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_RELEASE/bazelisk-linux-arm64" && \
    echo "BAZELISK_RELEASE=$BAZELISK_RELEASE" && echo "BAZELISK_URL=$BAZELISK_URL" && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate $BAZELISK_URL -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# have bazelisk download latest bazel
RUN bazel --version

# from source method
#ARG BAZEL_VERSION

#RUN mkdir bazel && \
#    cd bazel && \
#    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-dist.zip && \
#    unzip bazel-$BAZEL_VERSION-dist.zip && \
#    bash ./compile.sh
