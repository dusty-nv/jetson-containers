#!/usr/bin/env bash
set -ex

MODEL="${1:-gpt_350m}"

ENGINE_DIR="/data/models/tensorrt_llm/benchmarks/$MODEL"
CPP_BUILD_DIR="/opt/tensorrt_llm/cpp/build"

mkdir -p $ENGINE_DIR

echo "testing tensorrt_llm python benchmark ($MODEL)"

python3 /opt/tensorrt_llm/benchmarks/python/benchmark.py \
    -m $MODEL \
    --mode plugin \
    --batch_size "1" \
    --input_output_len "60,20;128,20" \
    --log_level verbose \
    --output_dir $ENGINE_DIR \
    --enable_cuda_graph \
    --strongly_typed

echo "tensorrt_llm python benchmark OK ($MODEL)"

if [ -d $CPP_BUILD_DIR ]; then
	
	cd $CPP_BUILD_DIR
	echo "testing tensorrt_llm cpp gptSessionBenchmark ($MODEL)"

	./benchmarks/gptManagerBenchmark --help

	./benchmarks/gptSessionBenchmark \
	    --engine_dir $ENGINE_DIR \
	    --batch_size "1" \
	    --input_output_len "60,20"

	echo "tensorrt_llm cpp gptSessionBenchmark OK ($MODEL)"
else
	echo "tensorrt_llm cpp build not found at $CPP_BUILD_DIR"
fi