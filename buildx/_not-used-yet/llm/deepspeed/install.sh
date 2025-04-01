#!/usr/bin/env bash
set -ex

pip3 install py-cpuinfo
python3 -m cpuinfo

apt-get update
apt-get install -y --no-install-recommends libaio-dev pdsh
rm -rf /var/lib/apt/lists/*
apt-get clean

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of DeepSpeed ${DEEPSPEED_VERSION} (branch=${DEEPSPEED_BRANCH})"
	exit 1
fi

pip3 install deepspeed==${DEEPSPEED_VERSION}
