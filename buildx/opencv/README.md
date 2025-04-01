# opencv

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`opencv:4.8.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `opencv` |
| &nbsp;&nbsp;&nbsp;Builds | [![`opencv-481_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/opencv-481_jp60.yml?label=opencv-481:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/opencv-481_jp60.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==35.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) |
| &nbsp;&nbsp;&nbsp;Dependants | [`audiocraft`](/packages/audio/audiocraft) [`deepstream`](/packages/deepstream) [`efficientvit`](/packages/vit/efficientvit) [`gstreamer`](/packages/gstreamer) [`jetson-inference`](/packages/jetson-inference) [`jetson-utils`](/packages/jetson-utils) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) [`nanoowl`](/packages/vit/nanoowl) [`ros:foxy-desktop`](/packages/ros) [`ros:foxy-ros-base`](/packages/ros) [`ros:foxy-ros-core`](/packages/ros) [`ros:galactic-desktop`](/packages/ros) [`ros:galactic-ros-base`](/packages/ros) [`ros:galactic-ros-core`](/packages/ros) [`ros:humble-desktop`](/packages/ros) [`ros:humble-ros-base`](/packages/ros) [`ros:humble-ros-core`](/packages/ros) [`ros:iron-desktop`](/packages/ros) [`ros:iron-ros-base`](/packages/ros) [`ros:iron-ros-core`](/packages/ros) [`ros:noetic-desktop`](/packages/ros) [`ros:noetic-ros-base`](/packages/ros) [`ros:noetic-ros-core`](/packages/ros) [`sam`](/packages/vit/sam) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`voicecraft`](/packages/audio/voicecraft) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/opencv:4.8.1-r36.2.0`](https://hub.docker.com/r/dustynv/opencv/tags) `(2023-12-07, 5.1GB)` |
| &nbsp;&nbsp;&nbsp;Notes | install or build OpenCV (with CUDA) from Jetson pip server |

| **`opencv:4.8.1-builder`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `opencv:builder` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==35.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | install or build OpenCV (with CUDA) from Jetson pip server |

| **`opencv:4.9.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==36.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | install or build OpenCV (with CUDA) from Jetson pip server |

| **`opencv:4.9.0-builder`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==36.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | install or build OpenCV (with CUDA) from Jetson pip server |

| **`opencv:4.5.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `opencv` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==32.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | install or build OpenCV (with CUDA) from Jetson pip server |

| **`opencv:4.5.0-builder`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `opencv:builder` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==32.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | install or build OpenCV (with CUDA) from Jetson pip server |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/opencv:4.8.1-r36.2.0`](https://hub.docker.com/r/dustynv/opencv/tags) | `2023-12-07` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/opencv:r32.7.1`](https://hub.docker.com/r/dustynv/opencv/tags) | `2023-12-06` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/opencv:r35.2.1`](https://hub.docker.com/r/dustynv/opencv/tags) | `2023-12-05` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/opencv:r35.3.1`](https://hub.docker.com/r/dustynv/opencv/tags) | `2023-08-29` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/opencv:r35.4.1`](https://hub.docker.com/r/dustynv/opencv/tags) | `2023-10-07` | `arm64` | `5.0GB` |

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
jetson-containers run $(autotag opencv)

# or explicitly specify one of the container images above
jetson-containers run dustynv/opencv:4.8.1-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/opencv:4.8.1-r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag opencv)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag opencv) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build opencv
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
