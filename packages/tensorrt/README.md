# tensorrt

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`tensorrt:8.6`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`tensorrt-86_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tensorrt-86_jp60.yml?label=tensorrt-86:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tensorrt-86_jp60.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==r36.*', '==cu122']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.deb`](Dockerfile.deb) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/tensorrt:8.6-r36.2.0`](https://hub.docker.com/r/dustynv/tensorrt/tags) `(2023-12-05, 6.7GB)` |

| **`tensorrt:10.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==r36.*', '==cu124']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.4`](/packages/cuda/cuda) [`cudnn:9.0`](/packages/cuda/cudnn) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.tar`](Dockerfile.tar) |

| **`tensorrt`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['<36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dependants | [`deepstream`](/packages/deepstream) [`efficientvit`](/packages/vit/efficientvit) [`jetson-inference`](/packages/jetson-inference) [`jetson-utils`](/packages/jetson-utils) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`onnxruntime:1.11`](/packages/onnxruntime) [`onnxruntime:1.11-builder`](/packages/onnxruntime) [`onnxruntime:1.16.3`](/packages/onnxruntime) [`onnxruntime:1.16.3-builder`](/packages/onnxruntime) [`onnxruntime:1.17`](/packages/onnxruntime) [`onnxruntime:1.17-builder`](/packages/onnxruntime) [`onnxruntime:1.19`](/packages/onnxruntime) [`onnxruntime:1.19-builder`](/packages/onnxruntime) [`optimum`](/packages/llm/optimum) [`piper-tts`](/packages/audio/piper-tts) [`ros:foxy-desktop`](/packages/ros) [`ros:foxy-ros-base`](/packages/ros) [`ros:foxy-ros-core`](/packages/ros) [`ros:galactic-desktop`](/packages/ros) [`ros:galactic-ros-base`](/packages/ros) [`ros:galactic-ros-core`](/packages/ros) [`ros:humble-desktop`](/packages/ros) [`ros:humble-ros-base`](/packages/ros) [`ros:humble-ros-core`](/packages/ros) [`ros:iron-desktop`](/packages/ros) [`ros:iron-ros-base`](/packages/ros) [`ros:iron-ros-core`](/packages/ros) [`ros:melodic-desktop`](/packages/ros) [`ros:melodic-ros-base`](/packages/ros) [`ros:melodic-ros-core`](/packages/ros) [`ros:noetic-desktop`](/packages/ros) [`ros:noetic-ros-base`](/packages/ros) [`ros:noetic-ros-core`](/packages/ros) [`sam`](/packages/vit/sam) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`tensorflow`](/packages/tensorflow) [`tensorflow2`](/packages/tensorflow) [`tensorrt_llm:0.10.dev0`](/packages/llm/tensorrt_llm) [`tensorrt_llm:0.10.dev0-builder`](/packages/llm/tensorrt_llm) [`tensorrt_llm:0.5`](/packages/llm/tensorrt_llm) [`tensorrt_llm:0.5-builder`](/packages/llm/tensorrt_llm) [`torch2trt`](/packages/pytorch/torch2trt) [`torch_tensorrt`](/packages/pytorch/torch_tensorrt) [`tritonserver`](/packages/tritonserver) [`wyoming-piper:master`](/packages/smart-home/wyoming/piper) [`xtts`](/packages/audio/xtts) [`zed`](/packages/hardware/zed) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/tensorrt:8.6-r36.2.0`](https://hub.docker.com/r/dustynv/tensorrt/tags) `(2023-12-05, 6.7GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/tensorrt:8.6-r36.2.0`](https://hub.docker.com/r/dustynv/tensorrt/tags) | `2023-12-05` | `arm64` | `6.7GB` |

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
jetson-containers run $(autotag tensorrt)

# or explicitly specify one of the container images above
jetson-containers run dustynv/tensorrt:8.6-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/tensorrt:8.6-r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag tensorrt)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag tensorrt) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build tensorrt
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
