# torch2trt

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`torch2trt`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`tensorrt`](/packages/cuda/tensorrt) |
| &nbsp;&nbsp;&nbsp;Dependants | [`clip_trt`](/packages/vit/clip_trt) [`l4t-pytorch`](/packages/ml/l4t/l4t-pytorch) [`local_llm`](/packages/llm/local_llm) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`pytorch:2.1-all`](/packages/pytorch) [`pytorch:2.2-all`](/packages/pytorch) [`pytorch:2.3-all`](/packages/pytorch) [`pytorch:2.3.1-all`](/packages/pytorch) [`pytorch:2.4-all`](/packages/pytorch) [`pytorch:2.5-all`](/packages/pytorch) [`pytorch:2.6-all`](/packages/pytorch) [`pytorch:2.7-all`](/packages/pytorch) [`pytorch:2.8-all`](/packages/pytorch) [`vscode:torch`](/packages/code/vscode) [`whisper_trt`](/packages/speech/whisper_trt) [`xtts`](/packages/speech/xtts) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/torch2trt:r32.7.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) `(2023-12-14, 1.1GB)`<br>[`dustynv/torch2trt:r35.2.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) `(2023-12-14, 5.5GB)`<br>[`dustynv/torch2trt:r35.3.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) `(2023-08-29, 5.5GB)`<br>[`dustynv/torch2trt:r35.4.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) `(2023-12-05, 5.5GB)`<br>[`dustynv/torch2trt:r36.4.0`](https://hub.docker.com/r/dustynv/torch2trt/tags) `(2024-10-04, 5.7GB)`<br>[`dustynv/torch2trt:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/torch2trt/tags) `(2025-03-03, 5.2GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/torch2trt:r32.7.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) | `2023-12-14` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/torch2trt:r35.2.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) | `2023-12-14` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/torch2trt:r35.3.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) | `2023-08-29` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/torch2trt:r35.4.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) | `2023-12-05` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/torch2trt:r36.4.0`](https://hub.docker.com/r/dustynv/torch2trt/tags) | `2024-10-04` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/torch2trt:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/torch2trt/tags) | `2025-03-03` | `arm64` | `5.2GB` |

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
jetson-containers run $(autotag torch2trt)

# or explicitly specify one of the container images above
jetson-containers run dustynv/torch2trt:r36.4.0-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/torch2trt:r36.4.0-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag torch2trt)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag torch2trt) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build torch2trt
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
