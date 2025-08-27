#!/usr/bin/env bash
set -ex

MODEL_NAME=${1:-"Llama-2-7b-hf"}
MODEL_PATH=${2:-"https://nvidia.box.com/shared/static/i3jtp8jdmdlsq4qkjof8v4muth8ar7fo.gz"}
MODEL_ROOT=${MODEL_ROOT:-"/data/models/mlc/dist"}

QUANTIZATION=${QUANTIZATION:-"q4f16_ft"}
QUANTIZATION_PATH="${MODEL_ROOT}/${MODEL_NAME}-${QUANTIZATION}"
SKIP_QUANTIZATION=${SKIP_QUANTIZATION:-"no"}

PROMPT=${PROMPT:-"/data/prompts/completion.json"}
CONV_TEMPLATE=${CONV_TEMPLATE:-"llama-2"}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-4096}
MAX_NUM_PROMPTS=${MAX_NUM_PROMPTS:-4}

USE_CACHE=${USE_CACHE:-1}
USE_SAFETENSORS=${USE_SAFETENSORS:-"auto"}

OUTPUT_CSV=${OUTPUT_CSV:-"/data/benchmarks/mlc.csv"}

quantize() # mlc_llm >= 0.1.1
{
	QUANTIZATION_LIB="$QUANTIZATION_PATH/$MODEL_NAME-$QUANTIZATION-cuda.so"
	
	if [ $SKIP_QUANTIZATION != "yes" ]; then
		python3 -m mlc_llm convert_weight $MODEL_PATH --quantization $QUANTIZATION --output $QUANTIZATION_PATH
		python3 -m mlc_llm gen_config $MODEL_PATH --quantization $QUANTIZATION --conv-template $CONV_TEMPLATE --context-window-size $MAX_CONTEXT_LEN --max-batch-size 1  --output $QUANTIZATION_PATH
		python3 -m mlc_llm compile $QUANTIZATION_PATH --device cuda --opt O3 --output $QUANTIZATION_LIB
	fi
	
	QUANTIZATION_LIB="--model-lib-path $QUANTIZATION_LIB"
}

quantize_legacy() # mlc_llm == 0.1.0
{
	if [ $SKIP_QUANTIZATION != "yes" ]; then
		MODEL_HF="$MODEL_ROOT/models/$MODEL_NAME"
		
		if [ ! -d "$MODEL_HF" ]; then
			ln -s $MODEL_PATH $MODEL_HF
		fi
		
		if [[ $USE_SAFETENSORS = "yes" ]]; then
			QUANT_FLAGS="--use-safetensors"
		fi
	
		python3 -m mlc_llm.build \
			--model $MODEL_NAME \
			--target cuda \
			--use-cuda-graph \
			--use-flash-attn-mqa \
			--use-cache $USE_CACHE \
			--quantization $QUANTIZATION \
			--artifact-path $MODEL_ROOT \
			--max-seq-len $MAX_CONTEXT_LEN \
			$QUANT_FLAGS
			
		if [ $? != 0 ]; then
			return 1
		fi
	else
		python3 -m mlc_llm.build --help
		
		if [ $? != 0 ]; then
			return 1
		fi
	fi
	
	QUANTIZATION_PATH="$QUANTIZATION_PATH/params"
}

if [[ $MODEL_PATH == http* ]]; then
	MODEL_EXTRACTED="$MODEL_ROOT/models/$MODEL_NAME"
	if [ ! -d "$MODEL_EXTRACTED" ]; then
		pushd $MODEL_ROOT/models
		MODEL_TAR="$MODEL_NAME.tar.gz"
		wget $WGET_FLAGS $MODEL_PATH -O $MODEL_TAR
		tar -xzvf $MODEL_TAR
		#rm $MODEL_TAR
		popd
	fi
	MODEL_PATH=$MODEL_EXTRACTED
fi

mkdir -p $MODEL_ROOT/models || echo "$MODEL_ROOT already exists"

quantize_legacy || quantize

python3 /test/benchmark.py \
	--model $QUANTIZATION_PATH $QUANTIZATION_LIB \
	--max-new-tokens 128 \
	--max-num-prompts 4 \
	--prompt $PROMPT \
	--save ${OUTPUT_CSV} 
	
