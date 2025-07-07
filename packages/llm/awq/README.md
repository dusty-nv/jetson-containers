# awq

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


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
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`awq:0.1.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `awq` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`triton`](/packages/ml/triton) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`opengl`](/packages/multimedia/opengl) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`deepspeed-kernels`](/packages/llm/deepspeed/deepspeed-kernels) [`deepspeed:0.9.5`](/packages/llm/deepspeed) [`decord2`](/packages/multimedia/decord) [`vila`](/packages/vlm/vila) [`sudonim`](/packages/llm/sudonim) |
| &nbsp;&nbsp;&nbsp;Dependants | [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/awq:r35.2.1`](https://hub.docker.com/r/dustynv/awq/tags) | `2023-12-14` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/awq:r35.3.1`](https://hub.docker.com/r/dustynv/awq/tags) | `2023-12-15` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/awq:r35.4.1`](https://hub.docker.com/r/dustynv/awq/tags) | `2023-12-12` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/awq:r36.2.0`](https://hub.docker.com/r/dustynv/awq/tags) | `2023-12-15` | `arm64` | `7.8GB` |
| &nbsp;&nbsp;[`dustynv/awq:r36.4.0`](https://hub.docker.com/r/dustynv/awq/tags) | `2025-02-02` | `arm64` | `5.3GB` |

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
jetson-containers run $(autotag awq)

# or explicitly specify one of the container images above
jetson-containers run dustynv/awq:r36.4.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/awq:r36.4.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag awq)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag awq) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build awq
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
