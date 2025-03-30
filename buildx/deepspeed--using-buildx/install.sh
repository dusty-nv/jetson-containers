#!/usr/bin/env bash
set -ex

# Install py-cpuinfo
pip3 install py-cpuinfo

# Display CPU info
python3 -m cpuinfo

# Update APT and install necessary packages
apt-get update
apt-get install -y --no-install-recommends libaio-dev pdsh

# Clean up APT cache
rm -rf /var/lib/apt/lists/*
apt-get clean

# Ensure numpy compatibility
pip3 install 'numpy<2'

# Check if DEEPSPEED_BRANCH is set and install it
if [ -n "$DEEPSPEED_BRANCH" ]; then
    pip3 install deepspeed==0.9.5
    echo "Building DeepSpeed (branch=$DEEPSPEED_BRANCH)"
    git clone --branch=$DEEPSPEED_BRANCH --depth=1 --recursive https://github.com/microsoft/DeepSpeed /opt/DeepSpeed
    cd /opt/DeepSpeed
    python3 setup.py install
else
    echo "DEEPSPEED_BRANCH is not set. Skipping DeepSpeed installation."
fi
