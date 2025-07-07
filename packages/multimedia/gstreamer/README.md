# gstreamer

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`gstreamer`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) |
| &nbsp;&nbsp;&nbsp;Dependants | [`deepstream`](/packages/cv/deepstream) [`dli-nano-ai`](/packages/ml/dli/dli-nano-ai) [`jetcam`](/packages/hw/jetcam) [`jetson-inference:foxy`](/packages/cv/jetson-inference) [`jetson-inference:galactic`](/packages/cv/jetson-inference) [`jetson-inference:humble`](/packages/cv/jetson-inference) [`jetson-inference:iron`](/packages/cv/jetson-inference) [`jetson-inference:jazzy`](/packages/cv/jetson-inference) [`jetson-inference:main`](/packages/cv/jetson-inference) [`jetson-utils:v1`](/packages/multimedia/jetson-utils) [`jetson-utils:v2`](/packages/multimedia/jetson-utils) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`local_llm`](/packages/llm/local_llm) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`nanoowl`](/packages/vit/nanoowl) [`sapiens`](/packages/vit/sapiens) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/gstreamer:r32.7.1`](https://hub.docker.com/r/dustynv/gstreamer/tags) `(2023-12-06, 0.7GB)`<br>[`dustynv/gstreamer:r35.2.1`](https://hub.docker.com/r/dustynv/gstreamer/tags) `(2023-09-07, 5.1GB)`<br>[`dustynv/gstreamer:r35.3.1`](https://hub.docker.com/r/dustynv/gstreamer/tags) `(2023-12-06, 5.1GB)`<br>[`dustynv/gstreamer:r35.4.1`](https://hub.docker.com/r/dustynv/gstreamer/tags) `(2023-10-07, 5.1GB)`<br>[`dustynv/gstreamer:r36.2.0`](https://hub.docker.com/r/dustynv/gstreamer/tags) `(2023-12-07, 5.4GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/gstreamer:r32.7.1`](https://hub.docker.com/r/dustynv/gstreamer/tags) | `2023-12-06` | `arm64` | `0.7GB` |
| &nbsp;&nbsp;[`dustynv/gstreamer:r35.2.1`](https://hub.docker.com/r/dustynv/gstreamer/tags) | `2023-09-07` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/gstreamer:r35.3.1`](https://hub.docker.com/r/dustynv/gstreamer/tags) | `2023-12-06` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/gstreamer:r35.4.1`](https://hub.docker.com/r/dustynv/gstreamer/tags) | `2023-10-07` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/gstreamer:r36.2.0`](https://hub.docker.com/r/dustynv/gstreamer/tags) | `2023-12-07` | `arm64` | `5.4GB` |

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
jetson-containers run $(autotag gstreamer)

# or explicitly specify one of the container images above
jetson-containers run dustynv/gstreamer:r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/gstreamer:r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag gstreamer)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag gstreamer) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build gstreamer
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
