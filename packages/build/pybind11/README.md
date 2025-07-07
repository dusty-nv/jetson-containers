# pybind11

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`pybind11`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `pybind11:global` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:compression-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:compression-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:data-tools-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:data-tools-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`jetson-inference:foxy`](/packages/cv/jetson-inference) [`jetson-inference:galactic`](/packages/cv/jetson-inference) [`jetson-inference:humble`](/packages/cv/jetson-inference) [`jetson-inference:iron`](/packages/cv/jetson-inference) [`jetson-inference:jazzy`](/packages/cv/jetson-inference) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`l4t-tensorflow:tf1`](/packages/ml/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/ml/l4t/l4t-tensorflow) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`ros:foxy-desktop`](/packages/robots/ros) [`ros:foxy-ros-base`](/packages/robots/ros) [`ros:foxy-ros-core`](/packages/robots/ros) [`ros:galactic-desktop`](/packages/robots/ros) [`ros:galactic-ros-base`](/packages/robots/ros) [`ros:galactic-ros-core`](/packages/robots/ros) [`ros:humble-desktop`](/packages/robots/ros) [`ros:humble-ros-base`](/packages/robots/ros) [`ros:humble-ros-core`](/packages/robots/ros) [`ros:iron-desktop`](/packages/robots/ros) [`ros:iron-ros-base`](/packages/robots/ros) [`ros:iron-ros-core`](/packages/robots/ros) [`ros:jazzy-desktop`](/packages/robots/ros) [`ros:jazzy-ros-base`](/packages/robots/ros) [`ros:jazzy-ros-core`](/packages/robots/ros) [`ros:noetic-desktop`](/packages/robots/ros) [`ros:noetic-ros-base`](/packages/robots/ros) [`ros:noetic-ros-core`](/packages/robots/ros) [`tensorflow2:2.16.1`](/packages/ml/tensorflow) [`tensorflow2:2.18.0`](/packages/ml/tensorflow) [`tensorflow2:2.19.0`](/packages/ml/tensorflow) [`tensorflow2:2.20.0`](/packages/ml/tensorflow) [`tensorflow2:2.21.0`](/packages/ml/tensorflow) [`tensorflow_graphics:2.18.0`](/packages/ml/tensorflow/graphics) [`tensorflow_graphics:2.19.0`](/packages/ml/tensorflow/graphics) [`tensorflow_graphics:2.20.0`](/packages/ml/tensorflow/graphics) [`tensorflow_text:2.18.0`](/packages/ml/tensorflow/text) [`tensorflow_text:2.19.0`](/packages/ml/tensorflow/text) [`tensorflow_text:2.20.0`](/packages/ml/tensorflow/text) [`zed:5.0-humble`](/packages/hw/zed) [`zed:5.0-jazzy`](/packages/hw/zed) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/pybind/pybind11 |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag pybind11)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host pybind11:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag pybind11)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag pybind11) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build pybind11
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
