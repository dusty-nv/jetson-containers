#!/usr/bin/env bash
# script for benchmarking various LLM transformers
set -e

# https://huggingface.co/docs/transformers/installation?highlight=transformers_cache#cache-setup
export TRANSFORMERS_CACHE="/test/data/transformers"  

benchmark_gpt()
{
	local model=$1
	local output="$TRANSFORMERS_CACHE/benchmarks.csv"
	
	python3 test_transformers.py --model=$model --output=$output --provider=tensorrt 
	python3 test_transformers.py --model=$model --output=$output --provider=tensorrt --fp16
	python3 test_transformers.py --model=$model --output=$output --provider=tensorrt --do-sample
	python3 test_transformers.py --model=$model --output=$output --provider=tensorrt --do-sample --fp16
	
	python3 test_transformers.py --model=$model --output=$output --provider=cuda
	python3 test_transformers.py --model=$model --output=$output --provider=cuda --do-sample
	
	python3 test_transformers.py --model=$model --output=$output --provider=cpu
	python3 test_transformers.py --model=$model --output=$output --provider=cpu --do-sample
}

benchmark_gpt "distilgpt2"
benchmark_gpt "MBZUAI/LaMini-GPT-124M"
benchmark_gpt "gpt2"
benchmark_gpt "gpt2-medium"
benchmark_gpt "gpt2-large"
benchmark_gpt "gpt2-xl"