#!/usr/bin/env bash
# launcher for jetson_containers/build.py (see docs/build.md)
ROOT="$(dirname $(dirname "$(readlink -f "$0")") )"
PYTHONPATH="$PYTHONPATH:$ROOT" python3 -m jetson_containers.build "$@"
