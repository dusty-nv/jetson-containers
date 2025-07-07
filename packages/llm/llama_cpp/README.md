# llama_cpp

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


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

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`llama_cpp:0.2.57`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`llama_cpp:0.2.70`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`llama_cpp:0.2.83`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`llama_cpp:0.2.90`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`llama_cpp:0.3.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`llama_cpp:0.3.2`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`llama_cpp:0.3.5`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/llama_cpp:0.3.5-r36.4.0`](https://hub.docker.com/r/dustynv/llama_cpp/tags) `(2025-01-09, 4.3GB)` |

| **`llama_cpp:0.3.6`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/llama_cpp:0.3.6-r36.4.0`](https://hub.docker.com/r/dustynv/llama_cpp/tags) `(2025-01-27, 3.9GB)` |

| **`llama_cpp:0.3.7`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/llama_cpp:0.3.7-r36.4.0`](https://hub.docker.com/r/dustynv/llama_cpp/tags) `(2025-01-29, 4.3GB)`<br>[`dustynv/llama_cpp:0.3.7-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) `(2025-03-03, 3.2GB)` |

| **`llama_cpp:0.3.8`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/llama_cpp:0.3.8-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) `(2025-04-24, 3.1GB)` |

| **`llama_cpp:0.3.9`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/llama_cpp:0.3.9-builder-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) `(2025-04-24, 3.5GB)`<br>[`dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) `(2025-04-24, 3.5GB)` |

| **`llama_cpp:0.4.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`llama_cpp:b5255`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/llama_cpp:b5255-r36.4-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) `(2025-05-02, 3.3GB)` |

| **`llama_cpp:b5833`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `llama_cpp` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-text-generation`](/packages/ml/l4t/l4t-text-generation) [`langchain`](/packages/rag/langchain) [`langchain:samples`](/packages/rag/langchain) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/llama_cpp:0.3.5-r36.4.0`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2025-01-09` | `arm64` | `4.3GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:0.3.6-r36.4.0`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2025-01-27` | `arm64` | `3.9GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:0.3.7-r36.4.0`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2025-01-29` | `arm64` | `4.3GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:0.3.7-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2025-03-03` | `arm64` | `3.2GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:0.3.8-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2025-04-24` | `arm64` | `3.1GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:0.3.9-builder-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2025-04-24` | `arm64` | `3.5GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2025-04-24` | `arm64` | `3.5GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:b5255-r36.4-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2025-05-02` | `arm64` | `3.3GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:b5283-r36.4-cu128-24.04`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2025-05-06` | `arm64` | `3.3GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:ggml-r35.2.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-12-05` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:ggml-r35.3.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-12-06` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:ggml-r35.4.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-12-19` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:ggml-r36.2.0`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-12-19` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:gguf-r35.2.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-12-15` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:gguf-r35.3.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-12-19` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:gguf-r35.4.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-12-15` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:gguf-r36.2.0`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-12-19` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:r35.2.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-08-29` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:r35.3.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-08-15` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:r35.4.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2024-09-12` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:r36.2.0`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2024-09-12` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/llama_cpp:r36.4.0`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2025-01-31` | `arm64` | `4.3GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag llama_cpp)

# or explicitly specify one of the container images above
jetson-containers run dustynv/llama_cpp:b5283-r36.4-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/llama_cpp:b5283-r36.4-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag llama_cpp)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag llama_cpp) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build llama_cpp
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
