#!/usr/bin/env bash
set -ex

LLAMA_EXAMPLES="/opt/tensorrt_llm/examples/llama"
TRT_LLM_MODELS="/data/models/tensorrt_llm"

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
	
	if [ ! -f $output_dir/*.safetensors ]; then
		python3 $LLAMA_EXAMPLES/convert_checkpoint.py \
			--model_dir $(huggingface-downloader meta-llama/Llama-2-7b-chat-hf) \
			--output_dir $output_dir \
			--dtype float16 \
			--ammo_quant_ckpt_path $(huggingface-downloader TheBloke/Llama-2-7B-Chat-GPTQ/model.safetensors) \
			--use_weight_only \
			--weight_only_precision int4_gptq \
			--group_size 128 \
			--per_group
	fi
	
	trtllm-build \
		--checkpoint_dir $output_dir \
		--output_dir $output_dir/engines \
		--gemm_plugin float16 \
		--log_level verbose
}


#llama_fp16
llama_gptq