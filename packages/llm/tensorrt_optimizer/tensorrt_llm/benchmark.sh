#!/usr/bin/env bash
set -ex

: ${MODEL:="gpt_350m"}  # llama_7b
: ${QUANTIZATION:="fp16"}  # fp8,fp8_gemm,fp8_kv_cache,int8_sq_per_tensor,int8_sq_per_token_channel,int8_weight_only,int4_weight_only,int4_weight_only_awq,int4_weight_only_gptq
: ${INPUT_OUTPUT_LEN:="60,20"}   # "60,20;128,20"
: ${ENABLE_PYTHON="on"}
: ${ENABLE_CPP="off"}

ENGINE_DIR="/data/models/tensorrt_llm/benchmarks/$MODEL-$QUANTIZATION"
mkdir -p $ENGINE_DIR

benchmark_python()
{
	echo "running tensorrt_llm python benchmark for $MODEL ($QUANTIZATION)"

	#uv pip uninstall nvidia-ml-py  # workaround for NVML 'not supported' errors on Jetson

	if [ -f $ENGINE_DIR/*.engine ]; then
		echo "TensorRT engine already exists under $ENGINE_DIR (skipping model builder)"
		PYTHON_FLAGS="--engine_dir $ENGINE_DIR $PYTHON_FLAGS"
	fi

	if [ $QUANTIZATION != "fp16" ]; then
		PYTHON_FLAGS="--quantization $QUANTIZATION $PYTHON_FLAGS"
	fi

	python3 /opt/tensorrt_llm/benchmarks/python/benchmark.py \
	    -m $MODEL \
	    --mode plugin \
	    --batch_size "1" \
	    --input_output_len $INPUT_OUTPUT_LEN \
	    --log_level verbose \
	    --output_dir $ENGINE_DIR \
	    --enable_cuda_graph \
	    --warm_up 2 \
	    --num_runs 3 \
	    --duration 10 \
	    --strongly_typed \
	    $PYTHON_FLAGS

	echo "done tensorrt_llm python benchmark for $MODEL ($QUANTIZATION)"
}

benchmark_cpp()
{
	echo "running tensorrt_llm python benchmark for $MODEL ($QUANTIZATION)"

	/opt/tensorrt_llm/cpp/build/benchmarks/gptSessionBenchmark \
	    --engine_dir $ENGINE_DIR \
	    --batch_size "1" \
	    --input_output_len $INPUT_OUTPUT_LEN

	echo "done tensorrt_llm python benchmark for $MODEL ($QUANTIZATION)"
}

if [ "$ENABLE_PYTHON" = "on" ]; then
	benchmark_python
fi

if [ "$ENABLE_CPP" = "on" ]; then
	benchmark_cpp
fi
