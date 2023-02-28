# Machine Learning Containers for Jetson and JetPack

Hosted on [NVIDIA GPU Cloud](https://ngc.nvidia.com/catalog/containers?orderBy=modifiedDESC&query=L4T&quickFilter=containers&filters=) (NGC) are the following Docker container images for machine learning on Jetson:

* [`l4t-ml`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-ml)
* [`l4t-pytorch`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch)
* [`l4t-tensorflow`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow)

The following ROS containers are also available, which can be pulled from [DockerHub](https://hub.docker.com/repository/docker/dustynv/ros) or built from source:

| Distro | Base | Desktop | PyTorch |
|----|:----:|:----:|:----:|
| ROS Melodic   | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=melodic)           | X | X |
| ROS Noetic    | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=noetic-ros-base)   | X | [`PyTorch`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=noetic-pytorch) |
| ROS2 Eloquent | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=eloquent)          | X | X |
| ROS2 Foxy     | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-ros-base)     | [`desktop`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-desktop) | [`PyTorch`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-pytorch) |
| ROS2 Galactic | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-ros-base) | [`desktop`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-desktop) | [`PyTorch`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-pytorch) |
| ROS2 Humble   | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-ros-base)   | [`desktop`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-desktop) | [`PyTorch`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-pytorch) |

The ROS distros that use Python3 have PyTorch-based containers, and some have ROS Desktop for JetPack 5.x.

## Pre-built Container Images

The following images can be pulled from NGC or DockerHub without needing to build the containers yourself:

|                                                                                     | L4T Version | Container Tag                                      |
|-------------------------------------------------------------------------------------|:-----------:|----------------------------------------------------|
| [`l4t-ml`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-ml)                 |   R35.2.1   | `nvcr.io/nvidia/l4t-ml:r35.2.1-py3`                |
|                                                                                     |   R35.1.0   | `nvcr.io/nvidia/l4t-ml:r35.1.0-py3`                |
|                                                                                     |   R34.1.1   | `nvcr.io/nvidia/l4t-ml:r34.1.1-py3`                |
|                                                                                     |   R34.1.0   | `nvcr.io/nvidia/l4t-ml:r34.1.0-py3`                |
|                                                                                     |   R32.7.1   | `nvcr.io/nvidia/l4t-ml:r32.7.1-py3`                |
|                                                                                     |   R32.6.1   | `nvcr.io/nvidia/l4t-ml:r32.6.1-py3`                |
|                                                                                     |   R32.5.0*  | `nvcr.io/nvidia/l4t-ml:r32.5.0-py3`                |
|                                                                                     |   R32.4.4   | `nvcr.io/nvidia/l4t-ml:r32.4.4-py3`                |
|                                                                                     |   R32.4.3   | `nvcr.io/nvidia/l4t-ml:r32.4.3-py3`                |
| [`l4t-pytorch`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch)       |   R35.2.1   | `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`    |
|                                                                                     |   R35.1.0   | `nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3`   |
|                                                                                     |   R35.1.0   | `nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3`   |
|                                                                                     |   R35.1.0   | `nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.11-py3`   |
|                                                                                     |   R34.1.1   | `nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.12-py3`   |
|                                                                                     |   R34.1.1   | `nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3`   |
|                                                                                     |   R34.1.0   | `nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3`   |
|                                                                                     |   R32.7.1   | `nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3`   |
|                                                                                     |   R32.7.1   | `nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.9-py3`    |
|                                                                                     |   R32.6.1   | `nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3`    |
|                                                                                     |   R32.6.1   | `nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.8-py3`    |
|                                                                                     |   R32.5.0*  | `nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3`    |
|                                                                                     |   R32.5.0*  | `nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.6-py3`    |
|                                                                                     |   R32.4.4   | `nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3`    |
|                                                                                     |   R32.4.3   | `nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3`    |
| [`l4t-tensorflow`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow) |   R35.2.1   | `nvcr.io/nvidia/l4t-tensorflow:r35.2.1-tf2.11-py3` |
|                                                                                     |   R35.1.0   | `nvcr.io/nvidia/l4t-tensorflow:r35.1.0-tf1.15-py3` |
|                                                                                     |   R35.1.0   | `nvcr.io/nvidia/l4t-tensorflow:r35.1.0-tf2.9-py3`  |
|                                                                                     |   R34.1.1   | `nvcr.io/nvidia/l4t-tensorflow:r34.1.1-tf1.15-py3` |
|                                                                                     |   R34.1.1   | `nvcr.io/nvidia/l4t-tensorflow:r34.1.1-tf2.8-py3`  |
|                                                                                     |   R34.1.0   | `nvcr.io/nvidia/l4t-tensorflow:r34.1.0-tf1.15-py3` |
|                                                                                     |   R34.1.0   | `nvcr.io/nvidia/l4t-tensorflow:r34.1.0-tf2.8-py3`  |
|                                                                                     |   R32.7.1   | `nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf1.15-py3` |
|                                                                                     |   R32.7.1   | `nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3`  |
|                                                                                     |   R32.6.1   | `nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf1.15-py3` |
|                                                                                     |   R32.6.1   | `nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3`  |
|                                                                                     |   R32.5.0*  | `nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf1.15-py3` |
|                                                                                     |   R32.5.0*  | `nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf2.3-py3`  |
|                                                                                     |   R32.4.4   | `nvcr.io/nvidia/l4t-tensorflow:r32.4.4-tf1.15-py3` |
|                                                                                     |   R32.4.4   | `nvcr.io/nvidia/l4t-tensorflow:r32.4.4-tf2.3-py3`  |
|                                                                                     |   R32.4.3   | `nvcr.io/nvidia/l4t-tensorflow:r32.4.3-tf1.15-py3` |
|                                                                                     |   R32.4.3   | `nvcr.io/nvidia/l4t-tensorflow:r32.4.3-tf2.2-py3`  |
| [`ROS Melodic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=melodic) <sup>(ros-base)</sup> |   R32.7.1   | `dustynv/ros:melodic-ros-base-l4t-r32.7.1`         |
|                                                                                     |   R32.6.1   | `dustynv/ros:melodic-ros-base-l4t-r32.6.1`         |
|                                                                                     |   R32.5.0*  | `dustynv/ros:melodic-ros-base-l4t-r32.5.0`         |
|                                                                                     |   R32.4.4   | `dustynv/ros:melodic-ros-base-l4t-r32.4.4`         |
| [`ROS Noetic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=noetic-ros-base) <sup>(ros-base)</sup> |   R35.2.1   | `dustynv/ros:noetic-ros-base-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:noetic-ros-base-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:noetic-ros-base-l4t-r34.1.1`          |
|                                                                                     |   R34.1.0   | `dustynv/ros:noetic-ros-base-l4t-r34.1.0`          |
|                                                                                     |   R32.7.1   | `dustynv/ros:noetic-ros-base-l4t-r32.7.1`          |
|                                                                                     |   R32.6.1   | `dustynv/ros:noetic-ros-base-l4t-r32.6.1`          |
|                                                                                     |   R32.5.0*  | `dustynv/ros:noetic-ros-base-l4t-r32.5.0`          |
|                                                                                     |   R32.4.4   | `dustynv/ros:noetic-ros-base-l4t-r32.4.4`          |
| [`ROS Noetic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=noetic-pytorch) <sup>(PyTorch)</sup> |   R35.2.1   | `dustynv/ros:noetic-pytorch-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:noetic-pytorch-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:noetic-pytorch-l4t-r34.1.1`          |
|                                                                                     |   R34.1.0   | `dustynv/ros:noetic-pytorch-l4t-r34.1.0`          |
|                                                                                     |   R32.7.1   | `dustynv/ros:noetic-pytorch-l4t-r32.7.1`          |
|                                                                                     |   R32.6.1   | `dustynv/ros:noetic-pytorch-l4t-r32.6.1`          |
|                                                                                     |   R32.5.0*  | `dustynv/ros:noetic-pytorch-l4t-r32.5.0`          |
|                                                                                     |   R32.4.4   | `dustynv/ros:noetic-pytorch-l4t-r32.4.4`          |
| [`ROS2 Eloquent`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=eloquent) <sup>(ros-base)</sup> |   R32.7.1   | `dustynv/ros:eloquent-ros-base-l4t-r32.7.1`        |
|                                                                                     |   R32.6.1   | `dustynv/ros:eloquent-ros-base-l4t-r32.6.1`        |
|                                                                                     |   R32.5.0*  | `dustynv/ros:eloquent-ros-base-l4t-r32.5.0`        |
|                                                                                     |   R32.4.4   | `dustynv/ros:eloquent-ros-base-l4t-r32.4.4`        |
| [`ROS2 Foxy`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-ros-base) <sup>(ros-base)</sup> |   R35.2.1   | `dustynv/ros:foxy-ros-base-l4t-r35.2.1`            |
|                                                                                     |   R35.1.0   | `dustynv/ros:foxy-ros-base-l4t-r35.1.0`            |
|                                                                                     |   R34.1.1   | `dustynv/ros:foxy-ros-base-l4t-r34.1.1`            |
|                                                                                     |   R34.1.0   | `dustynv/ros:foxy-ros-base-l4t-r34.1.0`            |
|                                                                                     |   R32.7.1   | `dustynv/ros:foxy-ros-base-l4t-r32.7.1`            |
|                                                                                     |   R32.6.1   | `dustynv/ros:foxy-ros-base-l4t-r32.6.1`            |
|                                                                                     |   R32.5.0*  | `dustynv/ros:foxy-ros-base-l4t-r32.5.0`            |
|                                                                                     |   R32.4.4   | `dustynv/ros:foxy-ros-base-l4t-r32.4.4`            |
| [`ROS2 Foxy`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-desktop) <sup>(desktop)</sup> |   R35.2.1   | `dustynv/ros:foxy-desktop-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:foxy-desktop-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:foxy-desktop-l4t-r34.1.1`          |
| [`ROS2 Foxy`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-pytorch) <sup>(PyTorch)</sup> |   R35.2.1   | `dustynv/ros:foxy-pytorch-l4t-r35.2.1`            |
|                                                                                     |   R35.1.0   | `dustynv/ros:foxy-pytorch-l4t-r35.1.0`            |
|                                                                                     |   R34.1.1   | `dustynv/ros:foxy-pytorch-l4t-r34.1.1`            |
|                                                                                     |   R34.1.0   | `dustynv/ros:foxy-pytorch-l4t-r34.1.0`            |
|                                                                                     |   R32.7.1   | `dustynv/ros:foxy-pytorch-l4t-r32.7.1`            |
|                                                                                     |   R32.6.1   | `dustynv/ros:foxy-pytorch-l4t-r32.6.1`            |
|                                                                                     |   R32.5.0*  | `dustynv/ros:foxy-pytorch-l4t-r32.5.0`            |
|                                                                                     |   R32.4.4   | `dustynv/ros:foxy-pytorch-l4t-r32.4.4`            |
| [`ROS2 Galactic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-ros-base) <sup>(ros-base)</sup> |   R35.2.1   | `dustynv/ros:galactic-ros-base-l4t-r35.2.1`        |
|                                                                                     |   R35.1.0   | `dustynv/ros:galactic-ros-base-l4t-r35.1.0`        |
|                                                                                     |   R34.1.1   | `dustynv/ros:galactic-ros-base-l4t-r34.1.1`        |
|                                                                                     |   R34.1.0   | `dustynv/ros:galactic-ros-base-l4t-r34.1.0`        |
|                                                                                     |   R32.7.1   | `dustynv/ros:galactic-ros-base-l4t-r32.7.1`        |
|                                                                                     |   R32.6.1   | `dustynv/ros:galactic-ros-base-l4t-r32.6.1`        |
|                                                                                     |   R32.5.0*  | `dustynv/ros:galactic-ros-base-l4t-r32.5.0`        |
|                                                                                     |   R32.4.4   | `dustynv/ros:galactic-ros-base-l4t-r32.4.4`        |
| [`ROS2 Galactic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-desktop) <sup>(desktop)</sup> |   R35.2.1   | `dustynv/ros:galactic-desktop-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:galactic-desktop-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:galactic-desktop-l4t-r34.1.1`        |
| [`ROS2 Galactic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-pytorch) <sup>(PyTorch)</sup> |   R35.2.1   | `dustynv/ros:galactic-pytorch-l4t-r35.2.1`        |
|                                                                                     |   R35.1.0   | `dustynv/ros:galactic-pytorch-l4t-r35.1.0`        |
|                                                                                     |   R34.1.1   | `dustynv/ros:galactic-pytorch-l4t-r34.1.1`        |
|                                                                                     |   R34.1.0   | `dustynv/ros:galactic-pytorch-l4t-r34.1.0`        |
|                                                                                     |   R32.7.1   | `dustynv/ros:galactic-pytorch-l4t-r32.7.1`        |
|                                                                                     |   R32.6.1   | `dustynv/ros:galactic-pytorch-l4t-r32.6.1`        |
|                                                                                     |   R32.5.0*  | `dustynv/ros:galactic-pytorch-l4t-r32.5.0`        |
|                                                                                     |   R32.4.4   | `dustynv/ros:galactic-pytorch-l4t-r32.4.4`        |
| [`ROS2 Humble`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-ros-base) <sup>(ros-base)</sup> |   R35.2.1   | `dustynv/ros:humble-ros-base-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:humble-ros-base-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:humble-ros-base-l4t-r34.1.1`          |
|                                                                                     |   R34.1.0   | `dustynv/ros:humble-ros-base-l4t-r34.1.0`          |
| [`ROS2 Humble`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-desktop) <sup>(desktop)</sup> |   R35.2.1   | `dustynv/ros:humble-desktop-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:humble-desktop-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:humble-desktop-l4t-r34.1.1`          |
| [`ROS2 Humble`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-pytorch) <sup>(PyTorch)</sup> |   R35.2.1   | `dustynv/ros:humble-pytorch-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:humble-pytorch-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:humble-pytorch-l4t-r34.1.1`          |
|                                                                                     |   R34.1.0   | `dustynv/ros:humble-pytorch-l4t-r34.1.0`          |

> **note:** the L4T R32.5.0 containers can be run on both JetPack 4.5 (L4T R32.5.0) and JetPack 4.5.1 (L4T R32.5.1)

To download and run one of these images, you can use the included run script from the repo:

``` bash
# L4T version in the container tag should match your L4T version
$ scripts/docker_run.sh -c nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
```

For other configurations, below are the instructions to build and test the containers using the included Dockerfiles.

## Docker Default Runtime

To enable access to the CUDA compiler (nvcc) during `docker build` operations, add `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json` configuration file before attempting to build the containers:

``` json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia"
}
```

You will then want to restart the Docker service or reboot your system before proceeding.

## Building the Containers

To rebuild the containers from a Jetson device running [JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) or newer, first clone this repo:

``` bash
$ git clone https://github.com/dusty-nv/jetson-containers
$ cd jetson-containers
```

Before proceeding, make sure you have set your [Docker Default Runtime](#docker-default-runtime) to `nvidia` as shown above.

### ML Containers

To build the ML containers (`l4t-pytorch`, `l4t-tensorflow`, `l4t-ml`), use [`scripts/docker_build_ml.sh`](scripts/docker_build_ml.sh) - along with an optional argument of which container(s) to build: 

``` bash
$ ./scripts/docker_build_ml.sh all        # build all: l4t-pytorch, l4t-tensorflow, and l4t-ml
$ ./scripts/docker_build_ml.sh pytorch    # build only l4t-pytorch
$ ./scripts/docker_build_ml.sh tensorflow # build only l4t-tensorflow
```

> You have to build `l4t-pytorch` and `l4t-tensorflow` to build `l4t-ml`, because it uses those base containers in the multi-stage build.

Note that the TensorFlow and PyTorch pip wheel installers for aarch64 are automatically downloaded in the Dockerfiles from the [Jetson Zoo](https://elinux.org/Jetson_Zoo).

### ROS Containers

To build the ROS containers, use [`scripts/docker_build_ros.sh`](scripts/docker_build_ros.sh) with the `--distro` option to specify the name of the ROS distro to build and `--package` to specify the ROS package to build (the default package is `ros_base`):

``` bash
$ ./scripts/docker_build_ros.sh --distro all   # build all ROS distros (default)
$ ./scripts/docker_build_ros.sh --distro foxy  # build only foxy (ros_base)
$ ./scripts/docker_build_ros.sh --distro foxy --package desktop  # build foxy desktop (on JetPack 5.x)
```

The package options are:  `ros_base`, `ros_core`, and `desktop` - note that the ROS2 Desktop packages only build on JetPack 5.x.  You can also specify `--with-pytorch` to build variants with support for PyTorch. 

## Run the Containers

To run ROS container, first you should get the container name , type the command which built container, if container has been built successfully, it will give your container name like bellow.

```bash
$ ./scripts/docker_build_ros.s --distro humble
 ... 
Successfully built ebc1d71f00f3
Successfully tagged ros:humble-ros-base-l4t-r35.1.0 # ros:humble-ros-base-l4t-r35.1.0 is the container name
```

Then, type

```bash
$ ./scripts/docker_run.sh -c ros:humble-ros-base-l4t-r35.1.0
```

to run the container.

## Testing the Containers

To run a series of automated tests on the packages installed in the containers, run the following from your `jetson-containers` directory:

``` bash
$ ./scripts/docker_test_ml.sh all        # test all: l4t-pytorch, l4t-tensorflow, and l4t-ml
$ ./scripts/docker_test_ml.sh pytorch    # test only l4t-pytorch
$ ./scripts/docker_test_ml.sh tensorflow # test only l4t-tensorflow
```

To test ROS:

``` bash
$ ./scripts/docker_test_ros.sh all       # test if the build of ROS all was successful: 'melodic', 'noetic', 'eloquent', 'foxy'
$ ./scripts/docker_test_ros.sh melodic   # test if the build of 'ROS melodic' was successful
$ ./scripts/docker_test_ros.sh noetic    # test if the build of 'ROS noetic' was successful
$ ./scripts/docker_test_ros.sh eloquent  # test if the build of 'ROS eloquent' was successful
$ ./scripts/docker_test_ros.sh foxy      # test if the build of 'ROS foxy' was successful
```

