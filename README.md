![a header for a software project about building containers for AI and machine learning](/docs/images/header.jpg)

# Machine Learning Containers for Jetson and JetPack

[![l4t-pytorch](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp51.yml?label=l4t-pytorch)](/packages/l4t/l4t-pytorch)  [![l4t-tensorflow](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-tensorflow-tf2_jp51.yml?label=l4t-tensorflow)](/packages/l4t/l4t-tensorflow) [![l4t-ml](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-ml_jp51.yml?label=l4t-ml)](/packages/l4t/l4t-ml) [![l4t-diffusion](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-diffusion_jp51.yml?label=l4t-diffusion)](/packages/l4t/l4t-diffusion) [![l4t-text-generation](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-text-generation_jp51.yml?label=l4t-text-generation)](/packages/l4t/l4t-text-generation)

Modular container build system that provides various [**AI/ML packages**](packages) for [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) :rocket::robot:

| | |
|---|---|
| **ML** | [`pytorch`](packages/pytorch) [`tensorflow`](packages/tensorflow) [`onnxruntime`](packages/onnxruntime) [`deepstream`](packages/deepstream) [`tritonserver`](packages/tritonserver) [`jupyterlab`](packages/jupyterlab) [`stable-diffusion`](packages/diffusion/stable-diffusion) |
| **LLM** | [`transformers`](packages/llm/transformers) [`text-generation-webui`](packages/llm/text-generation-webui) [`text-generation-inference`](packages/llm/text-generation-inference) [`exllama`](packages/llm/exllama) [`bitsandbytes`](packages/llm/bitsandbytes) [`awq`](packages/llm/awq) [`AutoGPTQ`](packages/llm/auto_gptq) [`GPTQ-for-LLaMa`](packages/llm/gptq-for-llama) [`optimum`](packages/llm/optimum) [`xformers`](packages/llm/xformers) [`nemo`](packages/nemo) |
| **L4T** | [`l4t-pytorch`](packages/l4t/l4t-pytorch) [`l4t-tensorflow`](packages/l4t/l4t-tensorflow) [`l4t-ml`](packages/l4t/l4t-ml) [`l4t-diffusion`](packages/l4t/l4t-diffusion) [`l4t-text-generation`](packages/l4t/l4t-text-generation) |
| **CUDA** | [`cupy`](packages/cupy) [`cuda-python`](packages/cuda-python) [`pycuda`](packages/pycuda) [`numba`](packages/numba) [`cudf`](packages/rapids/cudf) [`cuml`](packages/rapids/cuml) |
| **Robotics** | [`ros`](packages/ros) [`ros2`](packages/ros) [`opencv:cuda`](packages/opencv) [`realsense`](packages/realsense) [`zed`](packages/zed) |

See the [**`packages`**](packages) directory for the full list, including pre-built container images and CI/CD status for JetPack/L4T.

Using the included tools, you can easily combine packages together for building your own containers.  Want to run ROS2 with PyTorch and Transformers?  No problem - just do the [system setup](/docs/setup.md), and build it on your Jetson like this:

```bash
$ ./build.sh --name=my_container ros:humble-desktop pytorch transformers
```

There are shortcuts for running containers too - this will pull or build a [`l4t-pytorch`](packages/l4t/l4t-pytorch) image that's compatible:

```bash
$ ./run.sh $(./autotag l4t-pytorch)
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

If you look at any package's readme (like [`l4t-pytorch`](packages/l4t/l4t-pytorch)), it will have detailed instructions for running it's container.

## Documentation

* [Package List](/packages)
* [Package Definitions](/docs/packages.md)
* [System Setup](/docs/setup.md)
* [Building Containers](/docs/build.md)
* [Running Containers](/docs/run.md)

Looking for the old jetson-containers?   See the [`legacy`](https://github.com/dusty-nv/jetson-containers/tree/legacy) branch
