#!/usr/bin/env bash
set -e

cd /opt/faiss_lite/build
./test

cd ../
python3 benchmark.py