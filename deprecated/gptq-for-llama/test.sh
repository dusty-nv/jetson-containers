#!/usr/bin/env bash

cd /opt/GPTQ-for-LLaMa

python3 llama.py --help
python3 test_kernel.py
