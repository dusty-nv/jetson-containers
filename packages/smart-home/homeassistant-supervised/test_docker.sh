#!/usr/bin/env bash
# Docker access for Home Assistant Supervised

set -exo pipefail

echo "Testing Docker access..."

docker info
cat /etc/docker/daemon.json

echo "Docker access - OK"
