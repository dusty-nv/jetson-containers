
* TensorRT-LLM 0.5 from [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.5.0/) (found under `/opt/tensorrt-llm`)

> [!NOTE]  
> Only the v0.5.0 version of TensorRT-LLM is available on JetPack 6 with TensorRT 8.6 at this time.

Below, we adapt the [llama example](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.5.0/examples/llama#gptq) to run llama-2-7b with TensorRT-LLM on Jetson using GTPQ quantization for weights (INT4) and FP16 for activations (W4A16).

## Model Config

To simplify the commands below, create a .env file for the containers with the desired model info:

```bash
echo -e 'MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
ENGINE_DIR=/data/models/trt-llm/llama-2-7b-chat-gptq
HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN>
CUDA_MODULE_LOADING=LAZY' \
 > /tmp/llama-2-7b-chat-gptq.env 
 ```

* `MODEL_REPO` - HuggingFace repo ID of the original model (unquantized)
* `ENGINE_DIR` - where the quantized model and TensorRT engine is stored (`/data` is mounted to [`jetson-containers/data`](/data))
* `HUGGINGFACE_TOKEN` - user token of HuggingFace account with [access to llama2](/packages/llm/transformers/README.md#llama2)

You can change this later or create variations to switch the model being used.

## Quantization

There are many pre-quantized GPTQ models that can be downloaded from HuggingFace Hub.  They should have the following configuration:

* 4 bits
* group size 128
* act-order off

For example, this command will download [`TheBloke/Llama-2-7B-Chat-GPTQ`](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ) to a cached directory and then symbolically link to it:

```bash
./run.sh --env-file /tmp/llama-2-7b-chat-gptq.env \
  $(./autotag huggingface_hub) /bin/bash -c '\
    mkdir -p $ENGINE_DIR && \
    ln -s $(huggingface-downloader TheBloke/Llama-2-7B-Chat-GPTQ/model.safetensors) $ENGINE_DIR/model.safetensors'
```

See the [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.5.0/examples/llama#groupwise-quantization-awqgptq) for more information about supported quantization methods.

### gptq-for-llama

To perform the quantization yourself using [`gptq-for-llama`](/packages/llm/gptq-for-llama), run this (it can take a few hours on AGX Orin)

```bash
./run.sh --env-file /tmp/llama-2-7b-chat-gptq.env \
  --workdir=/opt/GPTQ-for-LLaMa \
  $(./autotag gptq-for-llama) /bin/bash -c '\
    mkdir -p $ENGINE_DIR && \
    python3 llama.py \
      $(huggingface-downloader $MODEL_REPO) c4 \
	 --wbits 4 \
	 --groupsize 128 \
	 --true-sequential \
	 --save_safetensors $ENGINE_DIR/model.safetensors'
```

You can then test inference in PyTorch first to confirm that the quantizatized model still produces coherent output:

```bash
./run.sh --env-file /tmp/llama-2-7b-chat-gptq.env \
  --workdir=/opt/GPTQ-for-LLaMa \
  $(./autotag gptq-for-llama) /bin/bash -c '\
    python3 llama_inference.py \
      $(huggingface-downloader $MODEL_REPO) \
	 --wbits 4 \
	 --groupsize 128 \
	 --load $ENGINE_DIR/model.safetensors \
	 --text "once upon a time,"'
```

## Build Engine

Then use llama [`build.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llama/build.py) from the examples to build the TensorRT engine:

```bash
./run.sh --env-file /tmp/llama-2-7b-chat-gptq.env \
  --workdir=/opt/tensorrt-llm/examples/llama \
  $(./autotag tensorrt-llm) /bin/bash -c '\
    python3 build.py \
      --model_dir $MODEL_REPO \
      --quant_ckpt_path $ENGINE_DIR/model.safetensors \
      --output_dir $ENGINE_DIR \
      --dtype float16 \
      --remove_input_padding \
      --use_gpt_attention_plugin float16 \
      --use_gemm_plugin float16 \
      --use_weight_only \
      --weight_only_precision int4_gptq \
      --per_group \
      --enable_context_fmha \
      --use_rmsnorm_plugin \
      --visualize \
      --log_level verbose'
```

This typically takes around ~7 minutes to complete on Jetson AGX Orin for 7B.

## Chat

To chat with the model interactively from the terminal:

```bash
./run.sh --env-file /tmp/llama-2-7b-chat-gptq.env \
  --workdir=/opt/tensorrt-llm/examples/llama \
  $(./autotag tensorrt-llm) /bin/bash -c '\
    python3 run_chat.py \
      --tokenizer_dir $(huggingface-downloader $MODEL_REPO) \
      --engine_dir $ENGINE_DIR \
      --max_output_len 512 \
      --streaming'
```

## Benchmark

To run the inferencing benchmark (with 128 tokens input, 128 tokens output)

```bash
./run.sh --env-file /tmp/llama-2-7b-chat-gptq.env \
  --workdir=/opt/tensorrt-llm/benchmarks/python\
  $(./autotag tensorrt-llm) /bin/bash -c '\
    python3 benchmark.py \
      --model llama_7b \
	 --mode plugin \
	 --batch_size 1 \
	 --input_output_len "128,128" \
	 --engine_dir $ENGINE_DIR \
	 --enable_cuda_graph \
	 --log_level info'
```
