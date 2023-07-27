# ros

Various ROS/ROS2 containers for JetPack.  These build ROS from source to run them on the needed versions of Ubuntu.

Supported ROS distros:   `melodic` `noetic` `foxy` `galactic` `humble` `iron`
<br>
Supported ROS packages:  `ros_base` `ros_core` `desktop`

<details open>
<summary><b>CONTAINERS</b></summary>
<br>

| **`ros:melodic-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T <34` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.melodic`](Dockerfile.ros.melodic) |
| &nbsp;&nbsp;&nbsp;Notes | ROS Melodic is for JetPack 4 only |

| **`ros:melodic-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T <34` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.melodic`](Dockerfile.ros.melodic) |
| &nbsp;&nbsp;&nbsp;Notes | ROS Melodic is for JetPack 4 only |

| **`ros:melodic-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T <34` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.melodic`](Dockerfile.ros.melodic) |
| &nbsp;&nbsp;&nbsp;Notes | ROS Melodic is for JetPack 4 only |

| **`ros:noetic-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.noetic`](Dockerfile.ros.noetic) |

| **`ros:noetic-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.noetic`](Dockerfile.ros.noetic) |

| **`ros:noetic-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.noetic`](Dockerfile.ros.noetic) |

| **`ros:foxy-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:foxy-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:foxy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:galactic-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:galactic-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:galactic-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:humble-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:humble-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:iron-ros-base`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:iron-ros-core`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| **`ros:iron-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

</details>

<details open>
<summary><b>CONTAINER IMAGES</b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/ros:eloquent-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:eloquent-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:eloquent-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:eloquent-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `6.0GB` |
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
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `0.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-04-18` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-slam-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `2.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-slam-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `2.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:foxy-slam-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `2.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.7GB` |
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
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `0.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-04-18` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:galactic-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `1.0GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `1.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-05-26` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-deepstream-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-10-03` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `0.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-05-26` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:humble-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-26` | `arm64` | `1.0GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-26` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-25` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-25` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-26` | `arm64` | `1.4GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-26` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-25` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-25` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-26` | `arm64` | `0.7GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-26` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-25` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:iron-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-25` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/ros:melodic-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:melodic-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:melodic-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:melodic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `1.0GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `1.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-04-18` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.8GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-deepstream-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-10-03` | `arm64` | `4.3GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-08-06` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2021-09-23` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-03-02` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-19` | `arm64` | `0.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-04-18` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2022-09-23` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-04-29` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-03-28` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/ros:noetic-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags) | `2023-05-02` | `arm64` | `5.1GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b>RUN CONTAINER</b></summary>
<br>

To start the container, you can use the [`run.sh`](/run.sh)/[`autotag`](/autotag) helpers or construct a full [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag ros)

# or explicitly specify one of the container images above
./run.sh dustynv/ros:iron-desktop-l4t-r35.1.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/ros:iron-desktop-l4t-r35.1.0
```
> <sup>[`run.sh`](/run.sh) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from DockerHub, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag ros)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag ros) my_app --abc xyz
```
You can pass any options to `run.sh` that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command it constructs before executing it.
</details>
<details open>
<summary><b>BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do this System Setup, then run:
```bash
./build.sh ros
```
The dependencies from above will be built into the container, and it'll be tested.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
