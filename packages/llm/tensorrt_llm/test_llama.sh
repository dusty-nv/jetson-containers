#!/usr/bin/env bash
set -ex

LLAMA_EXAMPLES="/opt/tensorrt_llm/examples/llama"
TRT_LLM_MODELS="/data/models/tensorrt_llm"

: "${FORCE_BUILD:=off}"

llama_fp16() 
{
	output_dir="$TRT_LLM_MODELS/llama-2-7b-chat-fp16"
	
	if [ ! -f $output_dir/*.safetensors ]; then
		python3 $LLAMA_EXAMPLES/convert_checkpoint.py \
			--model_dir $(huggingface-downloader meta-llama/Llama-2-7b-chat-hf) \
			--output_dir $output_dir \
			--dtype float16
	fi

	trtllm-build \
		--checkpoint_dir $output_dir \
		--output_dir $output_dir/engines \
		--gemm_plugin float16
}

llama_gptq() 
{
	output_dir="$TRT_LLM_MODELS/llama-2-7b-chat-gptq"
	engine_dir="$output_dir/engines"
	
	# --int8_kv_cache \
	
	if [ ! -f $output_dir/*.safetensors ] || [ $FORCE_BUILD = "on" ]; then
		python3 $LLAMA_EXAMPLES/convert_checkpoint.py \
			--model_dir $(huggingface-downloader meta-llama/Llama-2-7b-chat-hf) \
			--output_dir $output_dir \
			--dtype float16 \
			--modelopt_quant_ckpt_path $(huggingface-downloader TheBloke/Llama-2-7B-Chat-GPTQ/model.safetensors) \
			--use_weight_only \
			--weight_only_precision int4_gptq \
			--group_size 128 \
			--per_group
	fi
	
	if [ ! -f $engine_dir/*.engine ] || [ $FORCE_BUILD = "on" ]; then
	    trtllm-build \
		    --checkpoint_dir $output_dir \
		    --output_dir $engine_dir \
		    --gemm_plugin float16 \
		    --log_level verbose \
		    --max_batch_size 1 \
		    --max_num_tokens 4096 \
		    --max_input_len 4096 \
		    --max_output_len 128
    fi
    
    if false; then
        python3 $LLAMA_EXAMPLES/../run.py \
            --max_output_len=50 \
            --tokenizer_dir $(huggingface-downloader meta-llama/Llama-2-7b-chat-hf) \
            --engine_dir $engine_dir
    fi

    python3 /opt/tensorrt_llm/benchmarks/python/benchmark.py \
        -m llama_7b \
        --mode plugin \
        --batch_size 1 \
        --input_output_len "16,128;32,128;64,128;128,128;256,128;512,128;1024,128;2048,128;3072,128;3968,128" \
        --log_level verbose \
        --enable_cuda_graph \
        --warm_up 2 \
        --num_runs 3 \
        --duration 10 \
        --strongly_typed \
        --quantization int4_weight_only_gptq \
        --engine_dir $engine_dir
}


#llama_fp16
llama_gptq
