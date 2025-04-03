#!/usr/bin/env bash
# launcher for jetson_containers/build.py (see docs/build.md)
ROOT="$(dirname "$(readlink -f "$0")")"
VENV="$ROOT/venv"

if [ -d $VENV ]; then
  source $VENV/bin/activate
fi

PYTHONPATH="$PYTHONPATH:$ROOT" python3 -m jetson_containers.build "$@"
