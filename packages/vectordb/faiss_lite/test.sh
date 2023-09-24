#!/usr/bin/env bash
set -e

cd /opt/faiss_lite/build
./test

cd ../faiss_lite
python3 faiss_lite.py