# python

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`python`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `python3` |
| &nbsp;&nbsp;&nbsp;Builds | [![`python_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/python_jp51.yml?label=python:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/python_jp51.yml) [![`python_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/python_jp46.yml?label=python:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/python_jp46.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) |
| &nbsp;&nbsp;&nbsp;Dependants | [`auto-gptq`](/packages/llm/auto-gptq) [`awq`](/packages/llm/awq) [`bitsandbytes`](/packages/llm/bitsandbytes) [`cmake:pip`](/packages/cmake/cmake_pip) [`cuda-python`](/packages/cuda-python) [`cudf`](/packages/rapids/cudf) [`cuml`](/packages/rapids/cuml) [`cupy`](/packages/cupy) [`deepstream`](/packages/deepstream) [`exllama`](/packages/llm/exllama) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`gstreamer`](/packages/gstreamer) [`jupyterlab`](/packages/jupyterlab) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`nemo`](/packages/nemo) [`numba`](/packages/numba) [`numpy`](/packages/numpy) [`onnx`](/packages/onnx) [`onnxruntime`](/packages/onnxruntime) [`opencv`](/packages/opencv) [`opencv_builder`](/packages/opencv/opencv_builder) [`optimum`](/packages/llm/optimum) [`protobuf:apt`](/packages/protobuf/protobuf_apt) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) [`pycuda`](/packages/pycuda) [`pytorch:1.10`](/packages/pytorch) [`pytorch:1.11`](/packages/pytorch) [`pytorch:1.12`](/packages/pytorch) [`pytorch:1.13`](/packages/pytorch) [`pytorch:1.9`](/packages/pytorch) [`pytorch:2.0`](/packages/pytorch) [`pytorch:2.1`](/packages/pytorch) [`realsense`](/packages/realsense) [`ros:foxy-desktop`](/packages/ros) [`ros:foxy-ros-base`](/packages/ros) [`ros:foxy-ros-core`](/packages/ros) [`ros:galactic-desktop`](/packages/ros) [`ros:galactic-ros-base`](/packages/ros) [`ros:galactic-ros-core`](/packages/ros) [`ros:humble-desktop`](/packages/ros) [`ros:humble-ros-base`](/packages/ros) [`ros:humble-ros-core`](/packages/ros) [`ros:iron-desktop`](/packages/ros) [`ros:iron-ros-base`](/packages/ros) [`ros:iron-ros-core`](/packages/ros) [`ros:melodic-desktop`](/packages/ros) [`ros:melodic-ros-base`](/packages/ros) [`ros:melodic-ros-core`](/packages/ros) [`ros:noetic-desktop`](/packages/ros) [`ros:noetic-ros-base`](/packages/ros) [`ros:noetic-ros-core`](/packages/ros) [`rust`](/packages/rust) [`stable-diffusion`](/packages/diffusion/stable-diffusion) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tensorflow`](/packages/tensorflow) [`tensorflow2`](/packages/tensorflow) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui`](/packages/llm/text-generation-webui) [`torch2trt`](/packages/pytorch/torch2trt) [`torch_tensorrt`](/packages/pytorch/torch_tensorrt) [`torchaudio`](/packages/pytorch/torchaudio) [`torchvision`](/packages/pytorch/torchvision) [`transformers`](/packages/llm/transformers) [`tritonserver`](/packages/tritonserver) [`xformers`](/packages/llm/xformers) [`zed`](/packages/zed) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/python:r32.7.1`](https://hub.docker.com/r/dustynv/python/tags) `(2023-08-04, 0.4GB)`<br>[`dustynv/python:r35.2.1`](https://hub.docker.com/r/dustynv/python/tags) `(2023-08-04, 5.0GB)`<br>[`dustynv/python:r35.4.1`](https://hub.docker.com/r/dustynv/python/tags) `(2023-08-04, 4.9GB)` |
| &nbsp;&nbsp;&nbsp;Notes | installs core python3 packages and pip |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/python:r32.7.1`](https://hub.docker.com/r/dustynv/python/tags) | `2023-08-04` | `arm64` | `0.4GB` |
| &nbsp;&nbsp;[`dustynv/python:r35.2.1`](https://hub.docker.com/r/dustynv/python/tags) | `2023-08-04` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/python:r35.4.1`](https://hub.docker.com/r/dustynv/python/tags) | `2023-08-04` | `arm64` | `4.9GB` |

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
./run.sh $(./autotag python)

# or explicitly specify one of the container images above
./run.sh dustynv/python:r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/python:r35.4.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag python)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag python) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh python
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
