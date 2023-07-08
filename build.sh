#!/usr/bin/env bash

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#PYTHONPATH="$PYTHONPATH:$ROOT" python3 $ROOT/jetson_containers/build.py "$@"

PYTHONPATH="$PYTHONPATH:$ROOT" python3 -m jetson_containers.build "$@"
