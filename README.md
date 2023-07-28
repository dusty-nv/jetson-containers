# Machine Learning Containers for Jetson and JetPack

![NVIDIA](https://img.shields.io/static/v1?style=for-the-badge&message=NVIDIA&color=222222&logo=NVIDIA&logoColor=76B900&label=) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%26FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![ROS](https://img.shields.io/badge/ros-%230A0FF9.svg?style=for-the-badge&logo=ros&logoColor=white) 

Automated container build system provides [**AI/ML packages**](packages) for [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing).

| | |
|---|---|
| ML | [`pytorch`](packages/pytorch) [`tensorflow`](packages/tensorflow) [`onnxruntime`](packages/onnxruntime) [`deepstream`](packages/deepstream) [`tritonserver`](packages/tritonserver) [`nemo`](packages/nemo) [`jupyterlab`](packages/jupyterlab) |
| LLMs | [`transformers`](packages/llm/transformers) [`text-generation-webui`](packages/llm/text-generation-webui) [`exllama`](packages/llm/exllama) [`optimum`](packages/llm/optimum) [`awq`](packages/llm/awq) [`bitsandbytes`](packages/llm/bitsandbytes) [`AutoGPTQ`](packages/llm/auto-gptq) |
| CUDA | [`cupy`](packages/cupy) [`cuda-python`](packages/cuda-python) [`pycuda`](packages/pycuda) [`numba`](packages/numba) [`cudf`](packages/rapids/cudf) [`cuml`](packages/rapids/cuml) |
| Robotics | [`ros`](packages/ros) [`ros2`](packages/ros) [`opencv:cuda`](packages/opencv) [`realsense`](packages/realsense) [`zed`](packages/zed) |

See the [**`packages`**](packages) directory for the full list of packages, including pre-built container images.

Using the included tools, you can easily combine packages together for building your own containers.  Want to run ROS2 with PyTorch and Transformers?  No problem - do the [setup](README.md), and build it on your Jetson like this:

```bash
$ ./build.sh --name=my_container ros:humble-desktop pytorch transformers
```

Shortcuts are provided for running containers too - this will pull or build a compatible [`l4t-pytorch`](packages/l4t/l4t-pytorch) image:

```bash
$ ./run.sh $(autotag l4t-pytorch)  # find/build for your version of JetPack/L4T
```
> <sup>[`run.sh`](/run.sh) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

## Package Definitions

A package is one building block of a container - typically composed of a Dockerfile and optional configuration scripts.

You might notice that the Dockerfiles in this repo have special package metadata encoded in their header comments:

```dockerfile
#---
# name: pytorch
# alias: torch
# group: ml
# config: config.py
# depends: [python, numpy, onnx]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

...
```

The text between `#---` is YAML and is extracted by the build system.  Each package dict has the following keys:

| Key           |         Type         | Description                                                                             |
|---------------|:--------------------:|-----------------------------------------------------------------------------------------|
| `name`        |         `str`        | the name of the package                                                                 |
| `alias`       | `str` or `list[str]` | alternate names the package can be referred to by                                       |
| `build_args`  |        `dict`        | `ARG:VALUE` pairs that are `--build-args` to `docker build`                             |
| `build_flags` |         `str`        | additional options that get added to the `docker build` command                         |
| `config`      | `str` or `list[str]` | one or more config files to load (`.py`, `.json`, `.yml`, `.yaml`)                      |
| `depends`     | `str` or `list[str]` | list of packages that this package depends on, and will be built                        |
| `disabled`    | `bool`               | set to `true` for the package to be disabled                                            |
| `dockerfile`  | `str`                | filename of the Dockerfile (optional)                                                   |
| `docs`        | `str`                | text or markdown that is added to a package's auto-generated readme                     |
| `group`       | `str`                | optional group the package belongs to (e.g. `ml`, `llm`, `cuda`)                        |
| `notes`       | `str`                | brief one-line docs that are added to a package's readme table                          |
| `path`        | `str`                | path to the package's directory (automatically populated)                               |
| `requires`    | `str`                | the version(s) of L4T the package is compatible with (e.g. `>=35.2.1` for JetPack 5.1+) |
| `test`        | `str` or `list[str]` | one or more test commands/scripts to run (`.py`, `.sh`, or a shell command)             |

> these keys can all be accessed by any of the configuration methods below<br>
> any filenames or paths should be relative to the package's `path`<br>
> see the [Version Specifiers Specification](https://packaging.pypa.io/en/latest/specifiers.html) for valid syntax around `requires`

Packages can also include nested sub-packages (for example, all the [ROS variants](packages/ros)), which are typically generated in a config file.

### YAML

In lieu of having the package metadata right there in the Dockerfile header, packages can provide a separate YAML file (normally called `config.yaml` or `config.yml`) with the same information:

```yaml
name: pytorch
alias: torch
group: ml
config: config.py
depends: [python, numpy, onnx]
dockerfile: Dockerfile
test: test.py
```

This would be equivalent to having it encoded into the Dockerfile like above.

### JSON

Config files can also be provided in JSON format (normally called `config.json`).  The JSON and YAML configs typically get used when defining meta-containers that may not even have their own Dockerfiles, but exist solely as combinations of other packages - like [`l4t-pytorch`](packages/l4t/l4t-pytorch) does:

```json
{
    "l4t-pytorch": {
        "group": "ml",
        "depends": ["pytorch", "torchvision", "torchaudio", "torch2trt", "opencv", "pycuda"]
    }
}
```

You can define multiple packages/containers per config file, like how [`l4t-tensorflow`](packages/l4t/l4t-tensorflow) has versions for both TF1/TF2:

```json
{
    "l4t-tensorflow:tf1": {
        "group": "ml",
        "depends": ["tensorflow", "opencv", "pycuda"]
    },
    
    "l4t-tensorflow:tf2": {
        "group": "ml",
        "depends": ["tensorflow2", "opencv", "pycuda"]
    }
}
```

### Python

Python configuration scripts (normally called `config.py`) are the most expressive and get executed at the start of a build, and can dynamically set build parameters based on your environment and version of JetPack/L4T.  They have a global `package` dict added to their scope by the build system, which is used to configure the package:

```python
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

if L4T_VERSION.major >= 34:  
    MY_PACKAGE_VERSION = 'v5.0'  # on JetPack 5
else:                        
    MY_PACKAGE_VERSION = 'v4.0'  # on JetPack 4

package['build_args'] = {
    'MY_PACKAGE_VERSION': MY_PACKAGE_VERSION,
    'CUDA_ARCHITECTURES': ';'.join(CUDA_ARCHITECTURES),
}
```

This example sets build args in a Dockerfile, based on the version of JetPack/L4T that's running and the GPU architectures to compile for.  Typically the package's static settings remain in the Dockerfile header for the best visibility, while `config.py` sets the dynamic ones.


The [`jetson_containers`](jetson_containers) module exposes these [system variables](jetson_containers/l4t_version.py) that you can import and parameterize Dockerfiles off of:

| Name                 |                                       Type                                      | Description                                                  |
|----------------------|:-------------------------------------------------------------------------------:|--------------------------------------------------------------|
| `L4T_VERSION`        | [`packaging.version.Version`](https://packaging.pypa.io/en/latest/version.html) | version of L4T from `/etc/nv_tegra_release`                  |
| `JETPACK_VERSION`    | [`packaging.version.Version`](https://packaging.pypa.io/en/latest/version.html) | version of JetPack corresponding to L4T version              |
| `PYTHON_VERSION`     | [`packaging.version.Version`](https://packaging.pypa.io/en/latest/version.html) | version of Python (`3.6` or `3.8`)                           |
| `CUDA_VERSION`       | [`packaging.version.Version`](https://packaging.pypa.io/en/latest/version.html) | version of CUDA (under `/usr/local/cuda`)                    |
| `CUDA_ARCHITECTURES` |                                   `list[int]`                                   | NVCC GPU architectures to generate code for (e.g. `[72,87]`) |
| `SYSTEM_ARCH`        |                                      `str`                                      | `aarch64` or `x86_64`                                        |
| `LSB_RELEASE`        |                                      `str`                                      | `18.04` or `20.04`                                           |
| `LSB_CODENAME`       |                                      `str`                                      | `bionic` or `focal`                                          |

Of course, it being Python, you can perform basically any other system queries/configuration you want using Python's built-in libraries, including manipulating files used by the build context, ect.
 

<details>
<summary><h3>Legacy Documentation</h3></summary>

This project provides Dockerfiles, build scripts, and container images for machine learning on [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing):

* [`l4t-ml`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-ml)
* [`l4t-pytorch`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch)
* [`l4t-tensorflow`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow)

The following ROS containers are also available, which can be pulled from [DockerHub](https://hub.docker.com/repository/docker/dustynv/ros) or built from source:

| Distro | Base | Desktop | PyTorch |
|----|:----:|:----:|:----:|
| ROS Melodic   | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=melodic)           | X | X |
| ROS Noetic    | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=noetic-ros-base)   | X | [`PyTorch`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=noetic-pytorch) |
| ROS2 Foxy     | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-ros-base)     | [`desktop`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-desktop) | [`PyTorch`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-pytorch) |
| ROS2 Galactic | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-ros-base) | [`desktop`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-desktop) | [`PyTorch`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-pytorch) |
| ROS2 Humble   | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-ros-base)   | [`desktop`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-desktop) | [`PyTorch`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-pytorch) |
| ROS2 Iron     | [`ros-base`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=iron-ros-base)   | [`desktop`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=iron-desktop) | [`PyTorch`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=iron-pytorch) |

The PyTorch-based ROS containers also have the [jetson-inference](https://github.com/dusty-nv/jetson-inference) and [ros_deep_learning](https://github.com/dusty-nv/ros_deep_learning) packages installed.

## Pre-Built Container Images

The following images can be pulled from NGC or DockerHub without needing to build the containers yourself:

<details>
<summary>
<a href=https://ngc.nvidia.com/catalog/containers/nvidia:l4t-ml><b>l4t-ml</b></a> (<code>nvcr.io/nvidia/l4t-ml:r35.2.1-py3</code>)
</summary>
</br>

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

</details>
<details>
<summary>
<a href=https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch><b>l4t-pytorch</b></a> (<code>nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3</code>)
</summary>
</br>

|                                                                                     | L4T Version | Container Tag                                      |
|-------------------------------------------------------------------------------------|:-----------:|----------------------------------------------------|
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

</details>
<details>
<summary>
<a href=https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow><b>l4t-tensorflow</b></a> (<code>nvcr.io/nvidia/l4t-tensorflow:r35.2.1-tf2.11-py3</code>)
</summary>
</br>

|                                                                                     | L4T Version | Container Tag                                      |
|-------------------------------------------------------------------------------------|:-----------:|----------------------------------------------------|
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

</details>

#### ROS

<details>
<summary>
<a href=https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=melodic><b>ROS Melodic</b></a> (<code>dustynv/ros:melodic-ros-base-l4t-r32.7.1</code>)
</summary>
</br>

|                                                                                     | L4T Version | Container Tag                                      |
|-------------------------------------------------------------------------------------|:-----------:|----------------------------------------------------|
| [`ROS Melodic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=melodic) <sup>(ros-base)</sup> |   R32.7.1   | `dustynv/ros:melodic-ros-base-l4t-r32.7.1`         |
|                                                                                     |   R32.6.1   | `dustynv/ros:melodic-ros-base-l4t-r32.6.1`         |
|                                                                                     |   R32.5.0*  | `dustynv/ros:melodic-ros-base-l4t-r32.5.0`         |
|                                                                                     |   R32.4.4   | `dustynv/ros:melodic-ros-base-l4t-r32.4.4`         |

</details>
<details>
<summary>
<a href=https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=noetic><b>ROS Noetic</b></a> (<code>dustynv/ros:noetic-ros-base-l4t-r35.3.1</code>)
</summary>
</br>

|                                                                                     | L4T Version | Container Tag                                      |
|-------------------------------------------------------------------------------------|:-----------:|----------------------------------------------------|
| [`ROS Noetic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=noetic-ros-base) <sup>(ros-base)</sup> |   R35.3.1   | `dustynv/ros:noetic-ros-base-l4t-r35.3.1`          |
|                                                                                     |   R35.2.1   | `dustynv/ros:noetic-ros-base-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:noetic-ros-base-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:noetic-ros-base-l4t-r34.1.1`          |
|                                                                                     |   R34.1.0   | `dustynv/ros:noetic-ros-base-l4t-r34.1.0`          |
|                                                                                     |   R32.7.1   | `dustynv/ros:noetic-ros-base-l4t-r32.7.1`          |
|                                                                                     |   R32.6.1   | `dustynv/ros:noetic-ros-base-l4t-r32.6.1`          |
|                                                                                     |   R32.5.0*  | `dustynv/ros:noetic-ros-base-l4t-r32.5.0`          |
|                                                                                     |   R32.4.4   | `dustynv/ros:noetic-ros-base-l4t-r32.4.4`          |
| [`ROS Noetic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=noetic-pytorch) <sup>(PyTorch)</sup> |   R35.3.1   | `dustynv/ros:noetic-pytorch-l4t-r35.3.1`          |
|                                                                                     |   R35.2.1   | `dustynv/ros:noetic-pytorch-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:noetic-pytorch-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:noetic-pytorch-l4t-r34.1.1`          |
|                                                                                     |   R34.1.0   | `dustynv/ros:noetic-pytorch-l4t-r34.1.0`          |
|                                                                                     |   R32.7.1   | `dustynv/ros:noetic-pytorch-l4t-r32.7.1`          |
|                                                                                     |   R32.6.1   | `dustynv/ros:noetic-pytorch-l4t-r32.6.1`          |
|                                                                                     |   R32.5.0*  | `dustynv/ros:noetic-pytorch-l4t-r32.5.0`          |
|                                                                                     |   R32.4.4   | `dustynv/ros:noetic-pytorch-l4t-r32.4.4`          |

</details>

#### ROS2

<details>
<summary>
<a href=https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy><b>ROS2 Foxy</b></a> (<code>dustynv/ros:foxy-ros-base-l4t-r35.3.1</code>)
</summary>
</br>

|                                                                                     | L4T Version | Container Tag                                      |
|-------------------------------------------------------------------------------------|:-----------:|----------------------------------------------------|
| [`ROS2 Foxy`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-ros-base) <sup>(ros-base)</sup> |   R35.3.1   | `dustynv/ros:foxy-ros-base-l4t-r35.3.1`            |
|                                                                                     |   R35.2.1   | `dustynv/ros:foxy-ros-base-l4t-r35.2.1`            |
|                                                                                     |   R35.1.0   | `dustynv/ros:foxy-ros-base-l4t-r35.1.0`            |
|                                                                                     |   R34.1.1   | `dustynv/ros:foxy-ros-base-l4t-r34.1.1`            |
|                                                                                     |   R34.1.0   | `dustynv/ros:foxy-ros-base-l4t-r34.1.0`            |
|                                                                                     |   R32.7.1   | `dustynv/ros:foxy-ros-base-l4t-r32.7.1`            |
|                                                                                     |   R32.6.1   | `dustynv/ros:foxy-ros-base-l4t-r32.6.1`            |
|                                                                                     |   R32.5.0*  | `dustynv/ros:foxy-ros-base-l4t-r32.5.0`            |
|                                                                                     |   R32.4.4   | `dustynv/ros:foxy-ros-base-l4t-r32.4.4`            |
| [`ROS2 Foxy`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-desktop) <sup>(desktop)</sup> |   R35.3.1   | `dustynv/ros:foxy-desktop-l4t-r35.3.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:foxy-desktop-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:foxy-desktop-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:foxy-desktop-l4t-r34.1.1`          |
| [`ROS2 Foxy`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=foxy-pytorch) <sup>(PyTorch)</sup> |   R35.3.1   | `dustynv/ros:foxy-pytorch-l4t-r35.3.1`            |
|                                                                                     |   R35.2.1   | `dustynv/ros:foxy-pytorch-l4t-r35.2.1`            |
|                                                                                     |   R35.1.0   | `dustynv/ros:foxy-pytorch-l4t-r35.1.0`            |
|                                                                                     |   R34.1.1   | `dustynv/ros:foxy-pytorch-l4t-r34.1.1`            |
|                                                                                     |   R34.1.0   | `dustynv/ros:foxy-pytorch-l4t-r34.1.0`            |
|                                                                                     |   R32.7.1   | `dustynv/ros:foxy-pytorch-l4t-r32.7.1`            |
|                                                                                     |   R32.6.1   | `dustynv/ros:foxy-pytorch-l4t-r32.6.1`            |
|                                                                                     |   R32.5.0*  | `dustynv/ros:foxy-pytorch-l4t-r32.5.0`            |
|                                                                                     |   R32.4.4   | `dustynv/ros:foxy-pytorch-l4t-r32.4.4`            |

</details>
<details>
<summary>
<a href=https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic><b>ROS2 Galactic</b></a> (<code>dustynv/ros:galactic-ros-base-l4t-r35.3.1</code>)
</summary>
</br>

|                                                                                     | L4T Version | Container Tag                                      |
|-------------------------------------------------------------------------------------|:-----------:|----------------------------------------------------|
| [`ROS2 Galactic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-ros-base) <sup>(ros-base)</sup> |   R35.3.1   | `dustynv/ros:galactic-ros-base-l4t-r35.3.1`        |
|                                                                                     |   R35.2.1   | `dustynv/ros:galactic-ros-base-l4t-r35.2.1`        |
|                                                                                     |   R35.1.0   | `dustynv/ros:galactic-ros-base-l4t-r35.1.0`        |
|                                                                                     |   R34.1.1   | `dustynv/ros:galactic-ros-base-l4t-r34.1.1`        |
|                                                                                     |   R34.1.0   | `dustynv/ros:galactic-ros-base-l4t-r34.1.0`        |
|                                                                                     |   R32.7.1   | `dustynv/ros:galactic-ros-base-l4t-r32.7.1`        |
|                                                                                     |   R32.6.1   | `dustynv/ros:galactic-ros-base-l4t-r32.6.1`        |
|                                                                                     |   R32.5.0*  | `dustynv/ros:galactic-ros-base-l4t-r32.5.0`        |
|                                                                                     |   R32.4.4   | `dustynv/ros:galactic-ros-base-l4t-r32.4.4`        |
| [`ROS2 Galactic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-desktop) <sup>(desktop)</sup> |   R35.3.1   | `dustynv/ros:galactic-desktop-l4t-r35.3.1`          |
|                                                                                     |   R35.2.1   | `dustynv/ros:galactic-desktop-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:galactic-desktop-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:galactic-desktop-l4t-r34.1.1`        |
| [`ROS2 Galactic`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=galactic-pytorch) <sup>(PyTorch)</sup> |   R35.3.1   | `dustynv/ros:galactic-pytorch-l4t-r35.3.1`        |
|                                                                                     |   R35.2.1   | `dustynv/ros:galactic-pytorch-l4t-r35.2.1`        |
|                                                                                     |   R35.1.0   | `dustynv/ros:galactic-pytorch-l4t-r35.1.0`        |
|                                                                                     |   R34.1.1   | `dustynv/ros:galactic-pytorch-l4t-r34.1.1`        |
|                                                                                     |   R34.1.0   | `dustynv/ros:galactic-pytorch-l4t-r34.1.0`        |
|                                                                                     |   R32.7.1   | `dustynv/ros:galactic-pytorch-l4t-r32.7.1`        |
|                                                                                     |   R32.6.1   | `dustynv/ros:galactic-pytorch-l4t-r32.6.1`        |
|                                                                                     |   R32.5.0*  | `dustynv/ros:galactic-pytorch-l4t-r32.5.0`        |
|                                                                                     |   R32.4.4   | `dustynv/ros:galactic-pytorch-l4t-r32.4.4`        |

</details>
<details>
<summary>
<a href=https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble><b>ROS2 Humble</b></a> (<code>dustynv/ros:humble-ros-base-l4t-r35.3.1</code>)
</summary>
</br>

|                                                                                     | L4T Version | Container Tag                                      |
|-------------------------------------------------------------------------------------|:-----------:|----------------------------------------------------|
| [`ROS2 Humble`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-ros-base) <sup>(ros-base)</sup> |   R35.3.1   | `dustynv/ros:humble-ros-base-l4t-r35.3.1`          |
|                                                                                     |   R35.2.1   | `dustynv/ros:humble-ros-base-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:humble-ros-base-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:humble-ros-base-l4t-r34.1.1`          |
|                                                                                     |   R34.1.0   | `dustynv/ros:humble-ros-base-l4t-r34.1.0`          |
|                                                                                     |   R32.7.1   | `dustynv/ros:humble-ros-base-l4t-r32.7.1`          |
| [`ROS2 Humble`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-desktop) <sup>(desktop)</sup> |   R35.3.1   | `dustynv/ros:humble-desktop-l4t-r35.3.1`          |
|                                                                                     |   R35.2.1   | `dustynv/ros:humble-desktop-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:humble-desktop-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:humble-desktop-l4t-r34.1.1`          |
|                                                                                     |   R32.7.1   | `dustynv/ros:humble-desktop-l4t-r32.7.1`          |
| [`ROS2 Humble`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=humble-pytorch) <sup>(PyTorch)</sup> |   R35.3.1   | `dustynv/ros:humble-pytorch-l4t-r35.3.1`          |
|                                                                                     |   R35.2.1   | `dustynv/ros:humble-pytorch-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:humble-pytorch-l4t-r35.1.0`          |
|                                                                                     |   R34.1.1   | `dustynv/ros:humble-pytorch-l4t-r34.1.1`          |
|                                                                                     |   R34.1.0   | `dustynv/ros:humble-pytorch-l4t-r34.1.0`          |
|                                                                                     |   R32.7.1   | `dustynv/ros:humble-pytorch-l4t-r32.7.1`          |

</details>
<details>
<summary>
<a href=https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=iron><b>ROS2 Iron</b></a> (<code>dustynv/ros:iron-ros-base-l4t-r35.3.1</code>)
</summary>
</br>

|                                                                                     | L4T Version | Container Tag                                      |
|-------------------------------------------------------------------------------------|:-----------:|----------------------------------------------------|
| [`ROS2 Iron`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=iron-ros-base) <sup>(ros-base)</sup> |   R35.3.1   | `dustynv/ros:iron-ros-base-l4t-r35.3.1`          |
|                                                                                     |   R35.2.1   | `dustynv/ros:iron-ros-base-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:iron-ros-base-l4t-r35.1.0`          |
|                                                                                     |   R32.7.1   | `dustynv/ros:iron-ros-base-l4t-r32.7.1`          |
| [`ROS2 Iron`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=iron-desktop) <sup>(desktop)</sup> |   R35.3.1   | `dustynv/ros:iron-desktop-l4t-r35.3.1`          |
|                                                                                     |   R35.2.1   | `dustynv/ros:iron-desktop-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:iron-desktop-l4t-r35.1.0`          |
|                                                                                     |   R32.7.1   | `dustynv/ros:iron-desktop-l4t-r32.7.1`          |
| [`ROS2 Iron`](https://hub.docker.com/repository/registry-1.docker.io/dustynv/ros/tags?name=iron-pytorch) <sup>(PyTorch)</sup> |   R35.3.1   | `dustynv/ros:iron-pytorch-l4t-r35.3.1`          |
|                                                                                     |   R35.2.1   | `dustynv/ros:iron-pytorch-l4t-r35.2.1`          |
|                                                                                     |   R35.1.0   | `dustynv/ros:iron-pytorch-l4t-r35.1.0`          |
|                                                                                     |   R32.7.1   | `dustynv/ros:iron-pytorch-l4t-r32.7.1`          |

</details>

> **note:** L4T R32.x containers can run on other versions of R32.x (e.g. R32.7.1 containers can run on R32.7.2)<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; L4T R35 containers can run on other versions of R35 (e.g. R35.1.0 containers can run on R35.2.1)

To download and run one of these images, you can use the included run script from the repo:

``` bash
# L4T version in the container tag should match your L4T version
$ scripts/docker_run.sh -c nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
```

For other configurations, below are the instructions to build and test the containers using the included Dockerfiles.

## Building the Containers

To rebuild the containers from a Jetson device running [JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) or newer, first clone this repo:

``` bash
$ git clone https://github.com/dusty-nv/jetson-containers
$ cd jetson-containers
```

Before proceeding, make sure you have set your [Docker Default Runtime](#docker-default-runtime) to `nvidia` as shown below:

### Docker Default Runtime

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
$ ./scripts/docker_build_ros.sh --distro all     # build all ROS distros (default)
$ ./scripts/docker_build_ros.sh --distro humble  # build only humble (ros_base)
$ ./scripts/docker_build_ros.sh --distro humble --package desktop  # build humble desktop
```

The package options are:  `ros_base`, `ros_core`, and `desktop` - you can also specify `--with-pytorch` to build variants with support for PyTorch, [jetson-inference](https://github.com/dusty-nv/jetson-inference) and [ros_deep_learning](https://github.com/dusty-nv/ros_deep_learning). 

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

</details>

