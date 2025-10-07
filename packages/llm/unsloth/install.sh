#!/usr/bin/env bash
set -ex

apt-get update -y
apt-get install -y --no-install-recommends \
	libnuma-dev \
	libsndfile1 \
	libsndfile1-dev \
	libprotobuf-dev \
	libsm6 \
	libxext6 \
	libgl1

rm -rf /var/lib/apt/lists/*
apt-get clean

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of unsloth ${UNSLOTH_VERSION}"
	exit 1
fi

uv pip install \
	unsloth_zoo \
	cut_cross_entropy \
	unsloth==${UNSLOTH_VERSION}
