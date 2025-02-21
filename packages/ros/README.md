# ros

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

> [!NOTE]  
> <a href="https://github.com/NVIDIA-ISAAC-ROS"><b>NVIDIA Isaac ROS</b></a><br/>
> See [`github.com/NVIDIA-ISAAC-ROS`](https://github.com/NVIDIA-ISAAC-ROS) for optimized CV/AI packages and [NITROS](https://nvidia-isaac-ros.github.io/concepts/nitros/index.html) zero-copy transport.<br/>
> For Isaac ROS containers, use the [Isaac ROS Docker Development Environment](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_common/index.html#isaac-ros-docker-development-environment).

Below are ROS/ROS2 base containers for JetPack.  These build ROS from source to run them on the needed versions of Ubuntu.

Supported ROS distros: `melodic` `noetic` `foxy` `galactic` `humble` `iron` <br/>
Supported ROS packages: `ros_base` `ros_core` `desktop`

### Installing Add-on Packages

Since the ROS distributions included in these containers are built from source, you should not install additional ROS packages for them from apt - instead these should be built from source too.  There is a helper script for this [`/ros2_install.sh`](ros2_install.sh) which takes either a list of ROS package names or URL of a git repo, and builds/installs them in a ROS workspace:

```
# adds foxglove to ROS_ROOT (under /opt/ros)
/ros2_install.sh foxglove_bridge

# adds jetson-inference nodes under /ros2_workspace
ROS_WORKSPACE=/ros2_workspace /ros2_install.sh https://github.com/dusty-nv/ros_deep_learning
```

You can run this from the ROS2 container using a mounted directory for your workspace (where your compiled packages will be saved outside container), or via another Dockerfile using the ROS2 container as base (in which case your packages will be built into the container itself):

* [`/packages/ros/Dockerfile.ros2.extras`](/packages/ros/Dockerfile.ros2.extras)
* [`/packages/jetson-inference/Dockerfile.ros`](/packages/jetson-inference/Dockerfile.ros)

Examples of this being done you can find in the [`ros:humble-foxglove`](/packages/ros) and [`jetson-inference:humble`](/packages/jetson-inference) containers.


<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`ros:noetic-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-noetic-ros-base_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-ros-base_jp46.yml?label=ros-noetic-ros-base:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-ros-base_jp46.yml) [![`ros-noetic-ros-base_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-ros-base_jp51.yml?label=ros-noetic-ros-base:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-ros-base_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.noetic`](Dockerfile.ros.noetic) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:noetic-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) `(2021-08-06, 0.5GB)`<br>[`dustynv/ros:noetic-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2021-09-23, 0.5GB)`<br>[`dustynv/ros:noetic-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-03-02, 0.5GB)`<br>[`dustynv/ros:noetic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.6GB)`<br>[`dustynv/ros:noetic-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-04-18, 5.6GB)`<br>[`dustynv/ros:noetic-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-09-23, 5.6GB)`<br>[`dustynv/ros:noetic-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-04-29, 5.6GB)`<br>[`dustynv/ros:noetic-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.2GB)`<br>[`dustynv/ros:noetic-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-24, 5.2GB)`<br>[`dustynv/ros:noetic-ros-base-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-07, 5.2GB)` |

| **`ros:noetic-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-noetic-ros-core_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-ros-core_jp46.yml?label=ros-noetic-ros-core:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-ros-core_jp46.yml) [![`ros-noetic-ros-core_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-ros-core_jp51.yml?label=ros-noetic-ros-core:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-ros-core_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.noetic`](Dockerfile.ros.noetic) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:noetic-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.6GB)`<br>[`dustynv/ros:noetic-ros-core-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.2GB)`<br>[`dustynv/ros:noetic-ros-core-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-24, 5.2GB)`<br>[`dustynv/ros:noetic-ros-core-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-05, 5.2GB)` |

| **`ros:noetic-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-noetic-desktop_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-desktop_jp51.yml?label=ros-noetic-desktop:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-desktop_jp51.yml) [![`ros-noetic-desktop_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-noetic-desktop_jp46.yml?label=ros-noetic-desktop:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-noetic-desktop_jp46.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.noetic`](Dockerfile.ros.noetic) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:noetic-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.6GB)`<br>[`dustynv/ros:noetic-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 5.2GB)`<br>[`dustynv/ros:noetic-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-08-29, 5.2GB)`<br>[`dustynv/ros:noetic-desktop-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-07, 5.2GB)` |

| **`ros:foxy-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-foxy-ros-base_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-ros-base_jp51.yml?label=ros-foxy-ros-base:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-ros-base_jp51.yml) [![`ros-foxy-ros-base_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-ros-base_jp46.yml?label=ros-foxy-ros-base:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-ros-base_jp46.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:foxy-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) `(2021-08-06, 1.1GB)`<br>[`dustynv/ros:foxy-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2021-09-23, 1.1GB)`<br>[`dustynv/ros:foxy-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-03-02, 1.1GB)`<br>[`dustynv/ros:foxy-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.8GB)`<br>[`dustynv/ros:foxy-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-04-18, 5.9GB)`<br>[`dustynv/ros:foxy-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-09-23, 5.9GB)`<br>[`dustynv/ros:foxy-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-04-29, 5.9GB)`<br>[`dustynv/ros:foxy-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-09-07, 5.3GB)`<br>[`dustynv/ros:foxy-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.4GB)`<br>[`dustynv/ros:foxy-ros-base-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-07, 5.3GB)` |

| **`ros:foxy-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-foxy-ros-core_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-ros-core_jp46.yml?label=ros-foxy-ros-core:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-ros-core_jp46.yml) [![`ros-foxy-ros-core_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-ros-core_jp51.yml?label=ros-foxy-ros-core:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-ros-core_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:foxy-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.8GB)`<br>[`dustynv/ros:foxy-ros-core-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-08-29, 5.3GB)`<br>[`dustynv/ros:foxy-ros-core-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-24, 5.3GB)`<br>[`dustynv/ros:foxy-ros-core-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.3GB)` |

| **`ros:foxy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-foxy-desktop_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-desktop_jp46.yml?label=ros-foxy-desktop:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-desktop_jp46.yml) [![`ros-foxy-desktop_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-foxy-desktop_jp51.yml?label=ros-foxy-desktop:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-foxy-desktop_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-inference:foxy`](/packages/jetson-inference) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`ros:foxy-foxglove`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:foxy-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 1.1GB)`<br>[`dustynv/ros:foxy-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-09-23, 6.5GB)`<br>[`dustynv/ros:foxy-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-04-29, 6.4GB)`<br>[`dustynv/ros:foxy-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 5.9GB)`<br>[`dustynv/ros:foxy-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-24, 5.9GB)`<br>[`dustynv/ros:foxy-desktop-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-05, 5.9GB)` |

| **`ros:galactic-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-galactic-ros-base_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-ros-base_jp51.yml?label=ros-galactic-ros-base:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-ros-base_jp51.yml) [![`ros-galactic-ros-base_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-ros-base_jp46.yml?label=ros-galactic-ros-base:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-ros-base_jp46.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:galactic-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) `(2021-08-06, 0.8GB)`<br>[`dustynv/ros:galactic-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2021-09-23, 0.8GB)`<br>[`dustynv/ros:galactic-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-03-02, 0.8GB)`<br>[`dustynv/ros:galactic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.6GB)`<br>[`dustynv/ros:galactic-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-04-18, 5.6GB)`<br>[`dustynv/ros:galactic-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-09-23, 5.6GB)`<br>[`dustynv/ros:galactic-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-04-29, 5.6GB)`<br>[`dustynv/ros:galactic-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-08-29, 5.2GB)`<br>[`dustynv/ros:galactic-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-05, 5.2GB)`<br>[`dustynv/ros:galactic-ros-base-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.2GB)` |

| **`ros:galactic-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-galactic-ros-core_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-ros-core_jp51.yml?label=ros-galactic-ros-core:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-ros-core_jp51.yml) [![`ros-galactic-ros-core_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-ros-core_jp46.yml?label=ros-galactic-ros-core:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-ros-core_jp46.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:galactic-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.6GB)`<br>[`dustynv/ros:galactic-ros-core-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-08-29, 5.1GB)`<br>[`dustynv/ros:galactic-ros-core-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 5.2GB)`<br>[`dustynv/ros:galactic-ros-core-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-07, 5.2GB)` |

| **`ros:galactic-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-galactic-desktop_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-desktop_jp46.yml?label=ros-galactic-desktop:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-desktop_jp46.yml) [![`ros-galactic-desktop_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-galactic-desktop_jp51.yml?label=ros-galactic-desktop:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-galactic-desktop_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-inference:galactic`](/packages/jetson-inference) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`ros:galactic-foxglove`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:galactic-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 1.0GB)`<br>[`dustynv/ros:galactic-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-09-23, 6.2GB)`<br>[`dustynv/ros:galactic-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-04-29, 6.1GB)`<br>[`dustynv/ros:galactic-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-08-29, 5.7GB)`<br>[`dustynv/ros:galactic-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-24, 5.8GB)`<br>[`dustynv/ros:galactic-desktop-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.8GB)` |

| **`ros:humble-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-humble-ros-base_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-base_jp46.yml?label=ros-humble-ros-base:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-base_jp46.yml) [![`ros-humble-ros-base_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-base_jp51.yml?label=ros-humble-ros-base:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-base_jp51.yml) [![`ros-humble-ros-base_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-base_jp60.yml?label=ros-humble-ros-base:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-base_jp60.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:humble-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.7GB)`<br>[`dustynv/ros:humble-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-05-26, 5.6GB)`<br>[`dustynv/ros:humble-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-09-23, 5.6GB)`<br>[`dustynv/ros:humble-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-04-29, 5.6GB)`<br>[`dustynv/ros:humble-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-05, 5.2GB)`<br>[`dustynv/ros:humble-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.2GB)`<br>[`dustynv/ros:humble-ros-base-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-07, 5.2GB)`<br>[`dustynv/ros:humble-ros-base-l4t-r36.2.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 6.9GB)` |

| **`ros:humble-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-humble-ros-core_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-core_jp60.yml?label=ros-humble-ros-core:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-core_jp60.yml) [![`ros-humble-ros-core_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-core_jp46.yml?label=ros-humble-ros-core:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-core_jp46.yml) [![`ros-humble-ros-core_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-ros-core_jp51.yml?label=ros-humble-ros-core:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-ros-core_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:humble-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.7GB)`<br>[`dustynv/ros:humble-ros-core-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.2GB)`<br>[`dustynv/ros:humble-ros-core-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-24, 5.2GB)`<br>[`dustynv/ros:humble-ros-core-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-05, 5.2GB)`<br>[`dustynv/ros:humble-ros-core-l4t-r36.2.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 6.9GB)` |

| **`ros:humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-humble-desktop_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-desktop_jp60.yml?label=ros-humble-desktop:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-desktop_jp60.yml) [![`ros-humble-desktop_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-desktop_jp46.yml?label=ros-humble-desktop:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-desktop_jp46.yml) [![`ros-humble-desktop_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-humble-desktop_jp51.yml?label=ros-humble-desktop:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-humble-desktop_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-inference:humble`](/packages/jetson-inference) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`ros:humble-foxglove`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:humble-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-24, 1.0GB)`<br>[`dustynv/ros:humble-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2022-09-23, 6.2GB)`<br>[`dustynv/ros:humble-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-04-29, 6.2GB)`<br>[`dustynv/ros:humble-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-09-07, 5.8GB)`<br>[`dustynv/ros:humble-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.8GB)`<br>[`dustynv/ros:humble-desktop-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-07, 5.8GB)`<br>[`dustynv/ros:humble-desktop-l4t-r36.2.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 7.6GB)`<br>[`dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-11-14, 6.1GB)` |

| **`ros:iron-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-iron-ros-base_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-ros-base_jp46.yml?label=ros-iron-ros-base:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-ros-base_jp46.yml) [![`ros-iron-ros-base_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-ros-base_jp51.yml?label=ros-iron-ros-base:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-ros-base_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:iron-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.7GB)`<br>[`dustynv/ros:iron-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-05-26, 5.6GB)`<br>[`dustynv/ros:iron-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-09-07, 5.2GB)`<br>[`dustynv/ros:iron-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-05, 5.2GB)`<br>[`dustynv/ros:iron-ros-base-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 5.2GB)` |

| **`ros:iron-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-iron-ros-core_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-ros-core_jp46.yml?label=ros-iron-ros-core:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-ros-core_jp46.yml) [![`ros-iron-ros-core_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-ros-core_jp51.yml?label=ros-iron-ros-core:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-ros-core_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:iron-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-06, 0.7GB)`<br>[`dustynv/ros:iron-ros-core-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-08-29, 5.2GB)`<br>[`dustynv/ros:iron-ros-core-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.2GB)`<br>[`dustynv/ros:iron-ros-core-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-07, 5.2GB)` |

| **`ros:iron-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`ros-iron-desktop_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-desktop_jp46.yml?label=ros-iron-desktop:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-desktop_jp46.yml) [![`ros-iron-desktop_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/ros-iron-desktop_jp51.yml?label=ros-iron-desktop:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/ros-iron-desktop_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-inference:iron`](/packages/jetson-inference) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`ros:iron-foxglove`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ros:iron-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-24, 1.0GB)`<br>[`dustynv/ros:iron-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-05-26, 6.2GB)`<br>[`dustynv/ros:iron-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-09-07, 5.8GB)`<br>[`dustynv/ros:iron-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-10-24, 5.8GB)`<br>[`dustynv/ros:iron-desktop-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) `(2023-12-07, 5.8GB)` |

| **`ros:jazzy-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:jazzy-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-inference:jazzy`](/packages/jetson-inference) [`ros:jazzy-foxglove`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:foxy-foxglove`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) [`ros:foxy-desktop`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-inference:foxy`](/packages/jetson-inference) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2.extras`](Dockerfile.ros2.extras) |

| **`ros:galactic-foxglove`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) [`ros:galactic-desktop`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-inference:galactic`](/packages/jetson-inference) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2.extras`](Dockerfile.ros2.extras) |

| **`ros:humble-foxglove`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) [`ros:humble-desktop`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-inference:humble`](/packages/jetson-inference) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2.extras`](Dockerfile.ros2.extras) |

| **`ros:iron-foxglove`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) [`ros:iron-desktop`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-inference:iron`](/packages/jetson-inference) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2.extras`](Dockerfile.ros2.extras) |

| **`ros:jazzy-foxglove`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`numpy`](/packages/numpy) [`opencv:deb`](/packages/opencv) [`cmake`](/packages/build/cmake/cmake_pip) [`ros:jazzy-desktop`](/packages/ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-inference:jazzy`](/packages/jetson-inference) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2.extras`](Dockerfile.ros2.extras) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/ros:eloquent-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:eloquent-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:eloquent-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:eloquent-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-24` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-desktop-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-05` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-pytorch-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `1.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-pytorch-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `1.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-pytorch-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `1.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `1.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-pytorch-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-04-18` | `arm64` | `6.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-pytorch-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `6.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-04-18` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-09-07` | `arm64` | `5.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-07` | `arm64` | `5.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-core-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-08-29` | `arm64` | `5.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-core-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-24` | `arm64` | `5.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-core-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-slam-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `2.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-slam-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `2.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-slam-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `2.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `1.0GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-08-29` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-24` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-desktop-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-pytorch-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `1.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-pytorch-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `1.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-pytorch-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `1.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `1.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-pytorch-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-04-18` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-pytorch-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-04-18` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-08-29` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-05` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-core-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-08-29` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-core-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-core-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-24` | `arm64` | `1.0GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-09-07` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-07` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r36.2.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `7.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-11-14` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-llm-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2024-05-18` | `arm64` | `10.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-llm-r36.3.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2024-05-17` | `arm64` | `11.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `1.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-05-26` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-05-26` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-05` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r36.2.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `6.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-core-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-core-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-24` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-core-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-05` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-core-l4t-r36.2.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `6.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-24` | `arm64` | `1.0GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-26` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-09-07` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-24` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-desktop-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-26` | `arm64` | `1.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-26` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-25` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-25` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-26` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-09-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-05` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-base-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-core-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-08-29` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-core-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-core-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:melodic-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:melodic-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:melodic-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:melodic-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:melodic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:melodic-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-08-29` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-desktop-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `1.0GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `1.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-04-18` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-04-18` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-24` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-core-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-06` | `arm64` | `0.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-core-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-07` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-core-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-10-24` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-core-l4t-r35.4.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-12-05` | `arm64` | `5.2GB` |

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
jetson-containers run $(autotag ros)

# or explicitly specify one of the container images above
jetson-containers run dustynv/ros:humble-llm-r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/ros:humble-llm-r35.4.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag ros)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag ros) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build ros
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
