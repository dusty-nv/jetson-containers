#!/usr/bin/env bash

cd /opt/minigpt4.cpp/minigpt4

mem_capacity=$(grep MemTotal /proc/meminfo | awk '{print $2}')
echo "memory capacity:  $mem_capacity KB"

if [ $mem_capacity -le 8388608 ]; then
	python3 benchmark.py --max-new-tokens=32 --runs=1 \
	  $(huggingface-downloader --type=dataset maknee/minigpt4-7b-ggml/minigpt4-7B-f16.bin) \
	  $(huggingface-downloader --type=dataset maknee/ggml-vicuna-v0-quantized/ggml-vicuna-7B-v0-q5_k.bin)
else
	python3 benchmark.py --max-new-tokens=32 --runs=1 \
	  $(huggingface-downloader --type=dataset maknee/minigpt4-13b-ggml/minigpt4-13B-f16.bin) \
	  $(huggingface-downloader --type=dataset maknee/ggml-vicuna-v0-quantized/ggml-vicuna-13B-v0-q5_k.bin)	
fi