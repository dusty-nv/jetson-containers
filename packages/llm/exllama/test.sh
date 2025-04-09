#!/usr/bin/env bash
cd /opt/exllamav3

python3 test_inference.py --help

python3 test_inference.py -m $(huggingface-downloader TheBloke/Llama-2-7B-GPTQ) -p "Once upon a time,"