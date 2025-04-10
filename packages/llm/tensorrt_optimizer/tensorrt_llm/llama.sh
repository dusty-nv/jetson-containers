#!/usr/bin/env bash
set -ex

MODEL="meta-llama/Llama-2-7b-chat-hf"
QUANT="TheBloke/Llama-2-7B-Chat-GPTQ/model.safetensors"

LLAMA_EXAMPLES="/opt/TensorRT-LLM/examples/llama"
TRT_LLM_MODELS="/data/models/tensorrt_llm"

: "${FORCE_BUILD:=off}"


llama_fp16() 
{
	output_dir="$TRT_LLM_MODELS/$(basename $MODEL)-fp16"
	
	if [ ! -f $output_dir/*.safetensors ]; then
		python3 $LLAMA_EXAMPLES/convert_checkpoint.py \
			--model_dir $(huggingface-downloader $MODEL) \
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
	output_dir="$TRT_LLM_MODELS/$(basename $MODEL)-gptq"
	engine_dir="$output_dir/engines"
	
	if [ ! -f $output_dir/*.safetensors ] || [ $FORCE_BUILD = "on" ]; then
		python3 $LLAMA_EXAMPLES/convert_checkpoint.py \
			--model_dir $(huggingface-downloader $MODEL) \
			--output_dir $output_dir \
			--dtype float16 \
			--quant_ckpt_path $(huggingface-downloader $QUANT) \
			--use_weight_only \
			--weight_only_precision int4_gptq \
			--group_size 128 \
			--per_group
	fi
	
	if [ ! -f $engine_dir/*.engine ] || [ $FORCE_BUILD = "on" ]; then
	    trtllm-build \
		    --checkpoint_dir $output_dir \
		    --output_dir $engine_dir \
		    --gemm_plugin auto \
		    --log_level verbose \
		    --max_batch_size 1 \
		    --max_num_tokens 512 \
		    --max_seq_len 512 \
		    --max_input_len 128	    
    fi

    python3 $LLAMA_EXAMPLES/../run.py \
        --max_input_len=128 \
        --max_output_len=128 \
        --max_attention_window_size 256 \
        --max_tokens_in_paged_kv_cache=256 \
        --tokenizer_dir $(huggingface-downloader $MODEL) \
        --engine_dir $engine_dir

    python3 /opt/TensorRT-LLM/benchmarks/python/benchmark.py \
        -m dec \
        --engine_dir $engine_dir \
        --quantization int4_weight_only_gptq \
        --batch_size 1 \
        --input_output_len "16,128;32,128;64,128;128,128" \
        --log_level verbose \
        --enable_cuda_graph \
        --warm_up 2 \
        --num_runs 3 \
        --duration 10  
}

#llama_fp16
llama_gptq
