# exllama

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


This is using the https://github.com/jllllll/exllama fork of https://github.com/turboderp/exllama  

It's found under `/opt/exllama`, and the pip wheel is at `/opt/exllama-*.whl` and has been installed (with the CUDA kernels already built)

### Inference Benchmark

Substitute the GPTQ model from [HuggingFace Hub](https://huggingface.co/models?search=gptq) that you want to run (see [exllama-compatible models](https://github.com/turboderp/exllama/blob/master/doc/model_compatibility.md))

```bash
./run.sh --workdir=/opt/exllama $(./autotag exllama) /bin/bash -c \
  '/usr/bin/time -v python3 test_benchmark_inference.py --perf --validate -d $(huggingface-downloader TheBloke/Llama-2-7B-GPTQ)'
```
> If the model repository is private or requires authentication, add `--env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN>`

Memory usage:

| Model                                                                           | RAM (GB) | VRAM (GB) |
|---------------------------------------------------------------------------------|:--------:|:---------:|
| [`TheBloke/LLaMA-7b-GPTQ`](https://huggingface.co/TheBloke/LLaMA-7b-GPTQ)       |    3.4   |    5.2    |
| [`TheBloke/LLaMA-13b-GPTQ`](https://huggingface.co/TheBloke/LLaMA-13b-GPTQ)     |    3.5   |    9.2    |
| [`TheBloke/LLaMA-30b-GPTQ`](https://huggingface.co/TheBloke/LLaMA-30b-GPTQ)     |   3.25   |    20.2   |
| [`TheBloke/Llama-2-7B-GPTQ`](https://huggingface.co/TheBloke/Llama-2-7B-GPTQ)   |    3.5   |    5.2    |
| [`TheBloke/Llama-2-13B-GPTQ`](https://huggingface.co/TheBloke/Llama-2-13B-GPTQ) |    3.3   |    9.2    |
| [`TheBloke/Llama-2-70B-GPTQ`](https://huggingface.co/TheBloke/Llama-2-70B-GPTQ) |    3.1   |    35.5   |


<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`exllama`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`exllama_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/exllama_jp51.yml?label=exllama:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/exllama_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`huggingface_hub`](/packages/llm/huggingface_hub) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/exllama:r35.2.1`](https://hub.docker.com/r/dustynv/exllama/tags) `(2023-08-06, 5.4GB)`<br>[`dustynv/exllama:r35.3.1`](https://hub.docker.com/r/dustynv/exllama/tags) `(2023-08-06, 5.4GB)`<br>[`dustynv/exllama:r35.4.1`](https://hub.docker.com/r/dustynv/exllama/tags) `(2023-08-04, 5.4GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/exllama:r35.2.1`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-08-06` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/exllama:r35.3.1`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-08-06` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/exllama:r35.4.1`](https://hub.docker.com/r/dustynv/exllama/tags) | `2023-08-04` | `arm64` | `5.4GB` |

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
./run.sh $(./autotag exllama)

# or explicitly specify one of the container images above
./run.sh dustynv/exllama:r35.3.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/exllama:r35.3.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag exllama)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag exllama) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh exllama
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
