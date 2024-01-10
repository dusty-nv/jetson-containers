#!/usr/bin/env bash

set -e

cd /opt/text-generation-webui && python3 server.py --model-dir=/data/models/text-generation-webui --listen --verbose "$@"
