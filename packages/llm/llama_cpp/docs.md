
* llama.cpp from https://github.com/ggerganov/llama.cpp with CUDA enabled (found under `/opt/llama.cpp`)
* Python bindings from https://github.com/abetlen/llama-cpp-python (found under `/opt/llama-cpp-python`)

### Inference Benchmark

You can use llama.cpp's built-in [`main`](https://github.com/ggerganov/llama.cpp/tree/master/examples/main) tool to run GGML models (from [HuggingFace Hub](https://huggingface.co/models?search=ggml) or elsewhere)

```bash
./run.sh --workdir=/opt/llama.cpp/bin $(./autotag llama_cpp) /bin/bash -c \
   './main --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin) \
           --prompt "Once upon a time," \
           --n-predict 128 --ctx-size 192 --batch-size 192 \
           --n-gpu-layers 999 --threads $(nproc)'
```

```bash
./run.sh --workdir=/opt/llama.cpp/bin $(./autotag llama_cpp) /bin/bash -c '\
./main --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin) \
       --prompt "Once upon a time," \
       --n-predict 128 --ctx-size 192 --batch-size 192 \
       --n-gpu-layers 999 --threads $(nproc)'
```

```bash
./run.sh --workdir=/opt/llama.cpp/bin $(./autotag llama_cpp) /bin/bash -c \
'./main --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin) \
        --prompt "Once upon a time," \
        --n-predict 128 --ctx-size 192 --batch-size 192 \
        --n-gpu-layers 999 --threads $(nproc)'
```

```bash
./run.sh --workdir=/opt/llama.cpp/bin $(./autotag llama_cpp) /bin/bash -c \
 './main --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin) \
         --prompt "Once upon a time," \
         --n-predict 128 --ctx-size 192 --batch-size 192 \
         --n-gpu-layers 999 --threads $(nproc)'
```

> if you're trying to load Llama-2-70B, add the `--gqa 8` flag <br>
> the `--model` argument expects a `.bin` filename (typically the `*q4_0.bin` quantization is used)

To use the Python API and [`benchmark.py`](/packages/llm/llama_cpp/benchmark.py) instead:

```bash
./run.sh --workdir=/opt/llama.cpp/bin $(./autotag llama_cpp) /bin/bash -c \
   'python3 benchmark.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin) \
            --prompt "Once upon a time," \
            --n-predict 128 --ctx-size 192 --batch-size 192 \
            --n-gpu-layers 999 --threads $(nproc)'
```

```bash
./run.sh --workdir=/opt/llama.cpp/bin $(./autotag llama_cpp) /bin/bash -c '\
python3 benchmark.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin) \
          --prompt "Once upon a time," \
          --n-predict 128 --ctx-size 192 --batch-size 192 \
          --n-gpu-layers 999 --threads $(nproc)'
```

### Memory Usage

| Model                                                                           |          Quantization         | Memory (GB) |
|---------------------------------------------------------------------------------|:-----------------------------:|:-----------:|
| [`TheBloke/Llama-2-7B-GGML`](https://huggingface.co/TheBloke/Llama-2-7B-GGML)   |  `llama-2-7b.ggmlv3.q4_0.bin` |    5,268    |
| [`TheBloke/Llama-2-13B-GGML`](https://huggingface.co/TheBloke/Llama-2-13B-GGML) | `llama-2-13b.ggmlv3.q4_0.bin` |    8,609    |
| [`TheBloke/LLaMa-30B-GGML`](https://huggingface.co/TheBloke/LLaMa-30B-GGML)     | `llama-30b.ggmlv3.q4_0.bin`   |    19,045   |
| [`TheBloke/Llama-2-13B-GGML`](https://huggingface.co/TheBloke/Llama-2-70B-GGML) | `llama-2-70b.ggmlv3.q4_0.bin` |    37,655   |
