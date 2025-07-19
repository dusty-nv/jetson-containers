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

if [ "on" == "on" ]; then
	echo "Forcing build of mistral_common ${MISTRAL_COMMON_VERSION}"
	exit 1
fi

pip3 install \
	compressed-tensors \
	xgrammar \
	mistral_common==${MISTRAL_COMMON_VERSION}
