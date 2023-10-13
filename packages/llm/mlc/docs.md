
Container for [MLC LLM](https://github.com/mlc-ai/mlc-llm) project using Apache TVM Unity with CUDA, cuDNN, CUTLASS, FasterTransformer, and FlashAttention-2 kernels enabled.

### Model Quantization

First, download the original HF Transformers version of the model that you want to quantize with MLC, and symbolically link it under `/data/models/mlc/dist/models` so that MLC can find it properly:

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag mlc) /bin/bash -c '\
  ln -s $(huggingface-downloader meta-llama/Llama-2-7b-chat-hf) /data/models/mlc/dist/models/Llama-2-7b-chat-hf'
```

> [!NOTE]  
> If you are quantizing a Llava model, you need to change `"model_type": "llava"` to `"model_type": "llama"` in the original HF Transformers [`config.json`](https://huggingface.co/liuhaotian/llava-v1.5-7b/blob/main/config.json) version of the model (you can patch this locally after it's been downloaded)

Then perform W4A16 quantization on the model:

```bash
./run.sh $(./autotag mlc) \
  python3 -m mlc_llm.build \
    --model Llama-2-7b-chat-hf \
    --quantization q4f16_ft \
    --artifact-path /data/models/mlc/dist \
    --max-seq-len 4096 \
    --target cuda \
    --use-cuda-graph \
    --use-flash-attn-mqa
```

In this example, the quantized model and its runtime will be saved under `/data/models/mlc/dist/Llama-2-7b-chat-hf-q4f16_ft`

### Benchmarks

To benchmark the quantized model, run the [`benchmark.py`](benchmark.py) script:

```bash
./run.sh $(./autotag mlc) \
  python3 /opt/mlc-llm/benchmark.py \
    --model /data/models/mlc/ft_mqa/Llama-2-7b-chat-hf-q4f16_ft/params \
    --prompt /data/prompts/completion_16.json \
    --max-new-tokens 128
```

```
AVERAGE OVER 10 RUNS:
/data/models/mlc/ft_mqa/Llama-2-7b-chat-hf-q4f16_ft/params:  prefill_time 0.027 sec, prefill_rate 582.8 tokens/sec, decode_time 2.986 sec, decode_rate 42.9 tokens/sec
```

The `--prompt` file used controls the number of input tokens (context length) - there are generated prompt sequences under `/data/prompts` for up to 3968 tokens.  The `--max-new-tokens` argument specifies how many output tokens the model generates for each prompt.
