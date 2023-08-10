# llama_cpp

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


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

### Memory Usage

| Model                                                                           |          Quantization         | Memory (GB) |
|---------------------------------------------------------------------------------|:-----------------------------:|:-----------:|
| [`TheBloke/Llama-2-7B-GGML`](https://huggingface.co/TheBloke/Llama-2-7B-GGML)   |  `llama-2-7b.ggmlv3.q4_0.bin` |    5,268    |
| [`TheBloke/Llama-2-13B-GGML`](https://huggingface.co/TheBloke/Llama-2-13B-GGML) | `llama-2-13b.ggmlv3.q4_0.bin` |    8,609    |
| [`TheBloke/LLaMa-30B-GGML`](https://huggingface.co/TheBloke/LLaMa-30B-GGML)     | `llama-30b.ggmlv3.q4_0.bin`   |    19,045   |
| [`TheBloke/Llama-2-13B-GGML`](https://huggingface.co/TheBloke/Llama-2-70B-GGML) | `llama-2-70b.ggmlv3.q4_0.bin` |    37,655   |

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`llama_cpp`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`llama_cpp_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/llama_cpp_jp51.yml?label=llama_cpp:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/llama_cpp_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`huggingface_hub`](/packages/llm/huggingface_hub) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/llama_cpp:r35.2.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) `(2023-08-10, 5.2GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/llama_cpp:r35.2.1`](https://hub.docker.com/r/dustynv/llama_cpp/tags) | `2023-08-10` | `arm64` | `5.2GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag llama_cpp)

# or explicitly specify one of the container images above
./run.sh dustynv/llama_cpp:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/llama_cpp:r35.2.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag llama_cpp)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag llama_cpp) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh llama_cpp
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
