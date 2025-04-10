#---
# name: ffmpeg
# group: multimedia
# depends: [build-essential]
# notes: installs ffmpeg
# test: test.sh
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GOBIN=/opt/go/bin

# Install ffmpeg
RUN set -ex \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends \
	    ffmpeg \
	    libavcodec-dev \
	    libavfilter-dev \
	    libavformat-dev \
	    libavutil-dev \
	    libavdevice-dev \
	&& apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set Go env variables
ENV GOPATH=/opt/go \
	PATH=$PATH:$GOBIN

# Test ffmpeg version
RUN set -ex \
	&& ffmpeg -version \
	&& ffmpeg -encoders \
	&& ffmpeg -decoders
