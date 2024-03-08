#!/usr/bin/env bash

cd /opt/GPTQ-for-LLaMa

echo "testing gptq-for-llama..."

python3 llama.py --help
python3 test_kernel.py

echo "gptq-for-llama OK"
