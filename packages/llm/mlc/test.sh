#!/usr/bin/env bash
set -e

MODEL_ROOT="/data/models/mlc/dist"
MLC_USE_CACHE=${MLC_USE_CACHE:-1}

test_model()
{
	local MODEL_NAME=$1
	local MODEL_URL=$2
	local MODEL_TAR="$MODEL_NAME.tar.gz"
	local QUANTIZATION=${3:-q4f16_ft}
	
	if [ ! -d "$MODEL_ROOT/models/$MODEL_NAME" ]; then
		cd $MODEL_ROOT/models
		wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate $MODEL_URL -O $MODEL_TAR
		tar -xzvf $MODEL_TAR
		#rm $MODEL_TAR
	fi

	cd $MODEL_ROOT

	python3 -m mlc_llm.build \
		--model $MODEL_NAME \
		--target cuda \
		--use-cuda-graph \
		--use-flash-attn-mqa \
		--use-cache $MLC_USE_CACHE \
		--quantization $QUANTIZATION \
		--artifact-path $MODEL_ROOT \
		--max-seq-len 4096

	python3 /opt/mlc-llm/benchmark.py \
		--model ${MODEL_ROOT}/${MODEL_NAME}-${QUANTIZATION}/params \
		--max-new-tokens 128 \
		--max-num-prompts 4 \
		--prompt /data/prompts/completion.json
}

test_model "Llama-2-7b-hf" "https://nvidia.box.com/shared/static/i3jtp8jdmdlsq4qkjof8v4muth8ar7fo.gz"
