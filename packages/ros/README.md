# ros

Various ROS/ROS2 containers for JetPack.  These build ROS from source to run them on the needed versions of Ubuntu.

Supported ROS distros:   `melodic` `noetic` `foxy` `galactic` `humble` `iron`
</br>
Supported ROS packages:  `ros_base` `ros_core` `desktop`

| <font size="36">ros:melodic-ros-base</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T <34` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.melodic`](Dockerfile.ros.melodic) |
| &nbsp;&nbsp;&nbsp;Notes | ROS Melodic is for JetPack 4 only |

| <font size="36">ros:melodic-ros-core</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T <34` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.melodic`](Dockerfile.ros.melodic) |
| &nbsp;&nbsp;&nbsp;Notes | ROS Melodic is for JetPack 4 only |

| <font size="36">ros:melodic-desktop</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T <34` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.melodic`](Dockerfile.ros.melodic) |
| &nbsp;&nbsp;&nbsp;Notes | ROS Melodic is for JetPack 4 only |

| <font size="36">ros:noetic-ros-base</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.noetic`](Dockerfile.ros.noetic) |

| <font size="36">ros:noetic-ros-core</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.noetic`](Dockerfile.ros.noetic) |

| <font size="36">ros:noetic-desktop</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros.noetic`](Dockerfile.ros.noetic) |

| <font size="36">ros:foxy-ros-base</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:foxy-ros-core</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:foxy-desktop</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:galactic-ros-base</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:galactic-ros-core</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:galactic-desktop</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:humble-ros-base</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:humble-ros-core</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:humble-desktop</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:iron-ros-base</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:iron-ros-core</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |

| <font size="36">ros:iron-desktop</font> | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`cmake`](/packages/cmake/cmake_pip) [`numpy`](/packages/numpy) [`opencv`](/packages/opencv) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.ros2`](Dockerfile.ros2) |


<details open>
<summary><h3>Container Images</h3></summary>

- [`dustynv/ros:iron-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.2GB)`
- [`dustynv/ros:iron-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.3GB)`
- [`dustynv/ros:iron-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:iron-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.0GB)`
- [`dustynv/ros:iron-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.4GB)`
- [`dustynv/ros:iron-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.7GB)`
- [`dustynv/ros:iron-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.8GB)`
- [`dustynv/ros:iron-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.8GB)`
- [`dustynv/ros:iron-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.2GB)`
- [`dustynv/ros:iron-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.2GB)`
- [`dustynv/ros:iron-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.8GB)`
- [`dustynv/ros:iron-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.2GB)`
- [`dustynv/ros:humble-desktop-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.0GB)`
- [`dustynv/ros:humble-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.4GB)`
- [`dustynv/ros:humble-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.6GB)`
- [`dustynv/ros:galactic-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.4GB)`
- [`dustynv/ros:galactic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.6GB)`
- [`dustynv/ros:foxy-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.7GB)`
- [`dustynv/ros:foxy-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.9GB)`
- [`dustynv/ros:eloquent-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.8GB)`
- [`dustynv/ros:noetic-pytorch-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.3GB)`
- [`dustynv/ros:noetic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.6GB)`
- [`dustynv/ros:melodic-ros-base-l4t-r32.7.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.5GB)`
- [`dustynv/ros:humble-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.7GB)`
- [`dustynv/ros:humble-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.8GB)`
- [`dustynv/ros:humble-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.2GB)`
- [`dustynv/ros:galactic-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.7GB)`
- [`dustynv/ros:galactic-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.8GB)`
- [`dustynv/ros:galactic-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.1GB)`
- [`dustynv/ros:foxy-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.0GB)`
- [`dustynv/ros:foxy-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.1GB)`
- [`dustynv/ros:foxy-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.4GB)`
- [`dustynv/ros:noetic-desktop-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:noetic-pytorch-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.8GB)`
- [`dustynv/ros:noetic-ros-base-l4t-r35.3.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.1GB)`
- [`dustynv/ros:humble-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.2GB)`
- [`dustynv/ros:humble-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.3GB)`
- [`dustynv/ros:humble-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:galactic-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.1GB)`
- [`dustynv/ros:galactic-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.3GB)`
- [`dustynv/ros:galactic-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:foxy-desktop-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.4GB)`
- [`dustynv/ros:foxy-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.6GB)`
- [`dustynv/ros:foxy-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.9GB)`
- [`dustynv/ros:noetic-pytorch-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.3GB)`
- [`dustynv/ros:noetic-ros-base-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:humble-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.7GB)`
- [`dustynv/ros:humble-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.2GB)`
- [`dustynv/ros:humble-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.1GB)`
- [`dustynv/ros:galactic-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.7GB)`
- [`dustynv/ros:galactic-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.2GB)`
- [`dustynv/ros:galactic-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.1GB)`
- [`dustynv/ros:foxy-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.0GB)`
- [`dustynv/ros:foxy-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.5GB)`
- [`dustynv/ros:foxy-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.4GB)`
- [`dustynv/ros:noetic-desktop-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.5GB)`
- [`dustynv/ros:noetic-pytorch-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.1GB)`
- [`dustynv/ros:noetic-ros-base-l4t-r35.2.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.1GB)`
- [`dustynv/ros:noetic-ros-base-deepstream-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(4.3GB)`
- [`dustynv/ros:humble-ros-base-deepstream-l4t-r35.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.5GB)`
- [`dustynv/ros:humble-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.2GB)`
- [`dustynv/ros:humble-pytorch-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.1GB)`
- [`dustynv/ros:humble-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:galactic-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.2GB)`
- [`dustynv/ros:galactic-pytorch-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.1GB)`
- [`dustynv/ros:galactic-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:foxy-desktop-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.5GB)`
- [`dustynv/ros:foxy-pytorch-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.4GB)`
- [`dustynv/ros:foxy-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.9GB)`
- [`dustynv/ros:noetic-pytorch-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.0GB)`
- [`dustynv/ros:noetic-ros-base-l4t-r34.1.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:humble-pytorch-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.1GB)`
- [`dustynv/ros:humble-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:galactic-pytorch-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.1GB)`
- [`dustynv/ros:galactic-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:foxy-pytorch-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.4GB)`
- [`dustynv/ros:foxy-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.9GB)`
- [`dustynv/ros:noetic-pytorch-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(6.1GB)`
- [`dustynv/ros:noetic-ros-base-l4t-r34.1.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(5.6GB)`
- [`dustynv/ros:galactic-pytorch-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.3GB)`
- [`dustynv/ros:galactic-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.8GB)`
- [`dustynv/ros:foxy-slam-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(2.3GB)`
- [`dustynv/ros:foxy-pytorch-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.6GB)`
- [`dustynv/ros:foxy-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.1GB)`
- [`dustynv/ros:eloquent-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.8GB)`
- [`dustynv/ros:noetic-pytorch-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.0GB)`
- [`dustynv/ros:noetic-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.5GB)`
- [`dustynv/ros:melodic-ros-base-l4t-r32.6.1`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.5GB)`
- [`dustynv/ros:galactic-pytorch-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.3GB)`
- [`dustynv/ros:galactic-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.8GB)`
- [`dustynv/ros:foxy-slam-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(2.3GB)`
- [`dustynv/ros:foxy-pytorch-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.6GB)`
- [`dustynv/ros:foxy-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.1GB)`
- [`dustynv/ros:eloquent-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.8GB)`
- [`dustynv/ros:noetic-pytorch-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.1GB)`
- [`dustynv/ros:noetic-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.5GB)`
- [`dustynv/ros:melodic-ros-base-l4t-r32.5.0`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.5GB)`
- [`dustynv/ros:galactic-pytorch-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.3GB)`
- [`dustynv/ros:galactic-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.8GB)`
- [`dustynv/ros:foxy-slam-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(2.3GB)`
- [`dustynv/ros:foxy-pytorch-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.6GB)`
- [`dustynv/ros:foxy-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.1GB)`
- [`dustynv/ros:eloquent-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.8GB)`
- [`dustynv/ros:noetic-pytorch-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(1.1GB)`
- [`dustynv/ros:noetic-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.5GB)`
- [`dustynv/ros:melodic-ros-base-l4t-r32.4.4`](https://hub.docker.com/r/dustynv/ros/tags)  `arm64`  `(0.5GB)`
</details>

### Run Container
[`run.sh`](/run.sh) adds some default `docker run` args (like `--runtime nvidia`, mounts a [`/data`](/data) cache, and detects devices)
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag ros)

# or manually specify one of the container images above
./run.sh dustynv/ros:iron-desktop-l4t-r35.1.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/ros:iron-desktop-l4t-r35.1.0
```
To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag ros)
```
To start the container running a command, as opposed to the shell:
```bash
./run.sh $(./autotag ros) my_app --abc xyz
```
### Build Container
If you use [`autotag`](/autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do this System Setup, then run:
```bash
./build.sh ros
```
The dependencies from above will be built into the container, and it'll be tested.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
