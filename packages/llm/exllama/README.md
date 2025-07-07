# exllama

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


This package provides containers for both ExLlama and exllamav3:

* `exllama` container uses the https://github.com/jllllll/exllama fork of https://github.com/turboderp/exllama (installed under `/opt/exllama`)
* `exllama:v2` container uses https://github.com/turboderp/exllamav3 (installed under `/opt/exllamav3`)

Both loaders are also supported in the oobabooga [`text-generation-webui`](/packages/llm/text-generation-webui) container.

### Inference Benchmark

Substitute the GPTQ model from [HuggingFace Hub](https://huggingface.co/models?search=gptq) that you want to run (see [exllama compatible models](https://github.com/turboderp/exllama/blob/master/doc/model_compatibility.md))

```bash
./run.sh --workdir=/opt/exllama $(./autotag exllama) /bin/bash -c \
  'python3 test_benchmark_inference.py --perf --validate -d $(huggingface-downloader TheBloke/Llama-2-7B-GPTQ)'
```
> If the model repository is private or requires authentication, add `--env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN>`

### Memory Usage

| Model                                                                           | Memory (MB) |
|---------------------------------------------------------------------------------|:-----------:|
| [`TheBloke/Llama-2-7B-GPTQ`](https://huggingface.co/TheBloke/Llama-2-7B-GPTQ)   |    5,200    |
| [`TheBloke/Llama-2-13B-GPTQ`](https://huggingface.co/TheBloke/Llama-2-13B-GPTQ) |    9,135    |
| [`TheBloke/LLaMA-30b-GPTQ`](https://huggingface.co/TheBloke/LLaMA-30b-GPTQ)     |   20,206    |
| [`TheBloke/Llama-2-70B-GPTQ`](https://huggingface.co/TheBloke/Llama-2-70B-GPTQ) |   35,462    |


<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`exllama:0.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `exllama` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`huggingface_hub`](/packages/llm/huggingface_hub) [`triton`](/packages/ml/triton) [`torchvision`](/packages/pytorch/torchvision) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`flash-attention`](/packages/attention/flash-attention) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-text-generation`](/packages/ml/l4t/l4t-text-generation) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/exllama:r35.2.1`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-12-15` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/exllama:r35.3.1`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-12-11` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/exllama:r35.4.1`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-12-14` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/exllama:v1-r36.2.0`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-12-15` | `arm64` | `7.2GB` |
| &nbsp;&nbsp;[`dustynv/exllama:v2-r35.2.1`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-12-15` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/exllama:v2-r35.3.1`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-12-14` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/exllama:v2-r35.4.1`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-12-12` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/exllama:v2-r36.2.0`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-12-15` | `arm64` | `7.2GB` |

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
jetson-containers run $(autotag exllama)

# or explicitly specify one of the container images above
jetson-containers run dustynv/exllama:v1-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/exllama:v1-r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag exllama)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag exllama) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build exllama
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
