#!/usr/bin/env bash

cd /opt/minigpt4.cpp/minigpt4

python3 minigpt4_library.py \
  $(huggingface-downloader --type=dataset maknee/minigpt4-13b-ggml/minigpt4-13B-f16.bin) \
  $(huggingface-downloader --type=dataset maknee/ggml-vicuna-v0-quantized/ggml-vicuna-13B-v0-q5_k.bin)

#python3 minigpt4_library.py \
#  $(huggingface-downloader --type=dataset maknee/minigpt4-13b-ggml/minigpt4-13B-q4_0.bin) \
#  $(huggingface-downloader --type=dataset maknee/ggml-vicuna-v0-quantized/ggml-vicuna-13B-v0-q5_k.bin)
