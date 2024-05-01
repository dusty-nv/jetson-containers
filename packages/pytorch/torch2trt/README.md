# torch2trt

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`torch2trt`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`torch2trt_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/torch2trt_jp51.yml?label=torch2trt:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/torch2trt_jp51.yml) [![`torch2trt_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/torch2trt_jp46.yml?label=torch2trt:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/torch2trt_jp46.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch:2.2`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`tensorrt`](/packages/tensorrt) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`local_llm`](/packages/llm/local_llm) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`xtts`](/packages/audio/xtts) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/torch2trt:r32.7.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) `(2023-12-14, 1.1GB)`<br>[`dustynv/torch2trt:r35.2.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) `(2023-12-14, 5.5GB)`<br>[`dustynv/torch2trt:r35.3.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) `(2023-08-29, 5.5GB)`<br>[`dustynv/torch2trt:r35.4.1`](https://hub.docker.com/r/dustynv/torch2trt/tags) `(2023-12-05, 5.5GB)` |

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
jetson-containers run dustynv/torch2trt:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/torch2trt:r35.2.1
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
