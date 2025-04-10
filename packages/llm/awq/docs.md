
* AWQ from https://github.com/mit-han-lab/llm-awq (installed under `/opt/awq`)
* AWQ's CUDA kernels require a GPU with `sm_75` or newer (so for Jetson, Orin only)

### Quantization

Follow the instructions from https://github.com/mit-han-lab/llm-awq#usage to quantize your model of choice.  Or use [`awq/quantize.py`](/packages/llm/awq/quantize.py)

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag awq) /bin/bash -c \
  '/opt/awq/quantize.py --model=$(huggingface-downloader meta-llama/Llama-2-7b-hf) \
      --output=/data/models/awq/Llama-2-7b'
```

If you downloaded a model from the [AWQ Model Zoo](https://huggingface.co/datasets/mit-han-lab/awq-model-zoo) that already has the AWQ search results applied, you can load that with `--load_awq` and skip the search step (which can take a while and use lots of memory)

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag awq) /bin/bash -c \
  '/opt/awq/quantize.py --model=$(huggingface-downloader meta-llama/Llama-2-7b-hf) \
      --output=/data/models/awq/Llama-2-7b \
      --load_awq=/data/models/awq/Llama-2-7b/llama-2-7b-w4-g128.pt'
```

This process will save the model with the real quantized weights (to a file like `$OUTPUT/w4-g128-awq.pt`)

### Inference Benchmark

You can use the [`awq/benchmark.py`](/packages/llm/awq/quantize.py) tool to gather performance and memory measurements:

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag awq) /bin/bash -c \
  '/opt/awq/benchmark.py --model=$(huggingface-downloader meta-llama/Llama-2-7b-hf) \
      --quant=/data/models/awq/Llama-2-7b/w4-g128-awq.pt'
```

Make sure that you load the output from the quantization steps above with `--quant` (use the model that ends with `-awq.pt`)