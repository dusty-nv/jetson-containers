#!/usr/bin/env bash
set -ex

cansniffer -?

python3 -c 'import can; print(f"python-can version: {can.__version__}")'