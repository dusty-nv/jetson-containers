# opencv

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`opencv`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `opencv_cuda` |
| &nbsp;&nbsp;&nbsp;Builds | [![`opencv_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/opencv_jp46.yml?label=opencv:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/opencv_jp46.yml) [![`opencv_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/opencv_jp51.yml?label=opencv:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/opencv_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) |
| &nbsp;&nbsp;&nbsp;Dependants | [`deepstream`](/packages/deepstream) [`gstreamer`](/packages/gstreamer) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) [`ros:foxy-desktop`](/packages/ros) [`ros:foxy-ros-base`](/packages/ros) [`ros:foxy-ros-core`](/packages/ros) [`ros:galactic-desktop`](/packages/ros) [`ros:galactic-ros-base`](/packages/ros) [`ros:galactic-ros-core`](/packages/ros) [`ros:humble-desktop`](/packages/ros) [`ros:humble-ros-base`](/packages/ros) [`ros:humble-ros-core`](/packages/ros) [`ros:iron-desktop`](/packages/ros) [`ros:iron-ros-base`](/packages/ros) [`ros:iron-ros-core`](/packages/ros) [`ros:noetic-desktop`](/packages/ros) [`ros:noetic-ros-base`](/packages/ros) [`ros:noetic-ros-core`](/packages/ros) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/opencv:r32.7.1`](https://hub.docker.com/r/dustynv/opencv/tags) `(2023-08-12, 0.5GB)`<br>[`dustynv/opencv:r35.2.1`](https://hub.docker.com/r/dustynv/opencv/tags) `(2023-08-13, 5.0GB)`<br>[`dustynv/opencv:r35.3.1`](https://hub.docker.com/r/dustynv/opencv/tags) `(2023-08-13, 5.1GB)`<br>[`dustynv/opencv:r35.4.1`](https://hub.docker.com/r/dustynv/opencv/tags) `(2023-08-13, 5.0GB)` |
| &nbsp;&nbsp;&nbsp;Notes | install OpenCV (with CUDA) from binaries built by opencv_builder |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/opencv:r32.7.1`](https://hub.docker.com/r/dustynv/opencv/tags) | `2023-08-12` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/opencv:r35.2.1`](https://hub.docker.com/r/dustynv/opencv/tags) | `2023-08-13` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/opencv:r35.3.1`](https://hub.docker.com/r/dustynv/opencv/tags) | `2023-08-13` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/opencv:r35.4.1`](https://hub.docker.com/r/dustynv/opencv/tags) | `2023-08-13` | `arm64` | `5.0GB` |

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
./run.sh $(./autotag opencv)

# or explicitly specify one of the container images above
./run.sh dustynv/opencv:r35.3.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/opencv:r35.3.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag opencv)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag opencv) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh opencv
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
