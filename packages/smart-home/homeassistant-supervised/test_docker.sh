#!/usr/bin/env bash
# Docker access for Home Assistant Supervised

echo "Testing Docker access..."

docker info
cat /etc/docker/daemon.json

echo "Docker access - OK"
