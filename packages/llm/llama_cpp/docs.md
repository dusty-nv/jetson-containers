
* llama.cpp from https://github.com/ggerganov/llama.cpp with CUDA enabled (found under `/opt/llama.cpp`)
* Python bindings from https://github.com/abetlen/llama-cpp-python (found under `/opt/llama-cpp-python`)

> [!WARNING]  
> Starting with version 0.1.79, the model format has changed from GGML to GGUF.  Existing GGML models can be converted using the `convert-llama-ggmlv3-to-gguf.py` script in [`llama.cpp`](https://github.com/ggerganov/llama.cpp) (or you can often find the GGUF conversions on [HuggingFace Hub](https://huggingface.co/models?search=GGUF))

There are two branches of this container for backwards compatability:

* `llama_cpp:gguf` (the default, which tracks upstream master)
* `llama_cpp:ggml` (which still supports GGML model format)

There are a couple patches applied to the legacy GGML fork:

* fixed `__fp16` typedef in llama.h on ARM64 (use `half` with NVCC)
* parsing of BOS/EOS tokens (see https://github.com/ggerganov/llama.cpp/pull/1931)

### Inference Benchmark

You can use llama.cpp's built-in [`main`](https://github.com/ggerganov/llama.cpp/tree/master/examples/main) tool to run GGUF models (from [HuggingFace Hub](https://huggingface.co/models?search=gguf) or elsewhere)

```bash
./run.sh --workdir=/usr/local/bin $(./autotag llama_cpp) /bin/bash -c \
 './main --model $(huggingface-downloader TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_S.gguf) \
         --prompt "Once upon a time," \
         --n-predict 128 --ctx-size 192 --batch-size 192 \
         --n-gpu-layers 999 --threads $(nproc)'
```

> &gt; the `--model` argument expects a .gguf filename (typically the `Q4_K_S` quantization is used) <br>
> &gt; if you're trying to load Llama-2-70B, add the `--gqa 8` flag

To use the Python API and [`benchmark.py`](/packages/llm/llama_cpp/benchmark.py) instead:

```bash
./run.sh --workdir=/usr/local/bin $(./autotag llama_cpp) /bin/bash -c \
 'python3 benchmark.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_S.gguf) \
            --prompt "Once upon a time," \
            --n-predict 128 --ctx-size 192 --batch-size 192 \
            --n-gpu-layers 999 --threads $(nproc)'
```

To use a more contemporary model, such as `Llama-3.2-3B`, specify e.g. `unsloth/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf`.

### Memory Usage

| Model                                                                           |          Quantization         | Memory (MB) |
|---------------------------------------------------------------------------------|:-----------------------------:|:-----------:|
| [`TheBloke/Llama-2-7B-GGUF`](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)   | `llama-2-7b.Q4_K_S.gguf`      |    5,268    |
| [`TheBloke/Llama-2-13B-GGUF`](https://huggingface.co/TheBloke/Llama-2-13B-GGUF) | `llama-2-13b.Q4_K_S.gguf`     |    8,609    |
| [`TheBloke/LLaMA-30b-GGUF`](https://huggingface.co/TheBloke/LLaMA-30b-GGUF)     | `llama-30b.Q4_K_S.gguf`       |    19,045   |
| [`TheBloke/Llama-2-70B-GGUF`](https://huggingface.co/TheBloke/Llama-2-70B-GGUF) | `llama-2-70b.Q4_K_S.gguf`     |    37,655   |
