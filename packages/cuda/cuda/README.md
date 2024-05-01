# cuda

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`cuda:12.2`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`cuda-122_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cuda-122_jp60.yml?label=cuda-122:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cuda-122_jp60.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==35.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) |
| &nbsp;&nbsp;&nbsp;Dependants | [`cuda:12.2-samples`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`tensorrt:8.6`](/packages/tensorrt) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/cuda:12.2-r36.2.0`](https://hub.docker.com/r/dustynv/cuda/tags) `(2023-12-05, 3.4GB)`<br>[`dustynv/cuda:12.2-samples-r36.2.0`](https://hub.docker.com/r/dustynv/cuda/tags) `(2023-12-07, 4.8GB)` |

| **`cuda:12.4`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==36.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) |
| &nbsp;&nbsp;&nbsp;Dependants | [`cuda:12.4-samples`](/packages/cuda/cuda) [`cudnn:9.0`](/packages/cuda/cudnn) [`tensorrt:10.0`](/packages/tensorrt) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`cuda:12.2-samples`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`cuda-122-samples_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cuda-122-samples_jp60.yml?label=cuda-122-samples:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cuda-122-samples_jp60.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==35.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.samples`](Dockerfile.samples) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/cuda:12.2-samples-r36.2.0`](https://hub.docker.com/r/dustynv/cuda/tags) `(2023-12-07, 4.8GB)` |
| &nbsp;&nbsp;&nbsp;Notes | CUDA samples from https://github.com/NVIDIA/cuda-samples installed under /opt/cuda-samples |

| **`cuda:12.4-samples`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==36.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.4`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.samples`](Dockerfile.samples) |
| &nbsp;&nbsp;&nbsp;Notes | CUDA samples from https://github.com/NVIDIA/cuda-samples installed under /opt/cuda-samples |

| **`cuda:11.8`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==35.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) |
| &nbsp;&nbsp;&nbsp;Dependants | [`cuda:11.8-samples`](/packages/cuda/cuda) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`cuda:11.8-samples`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==35.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:11.8`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.samples`](Dockerfile.samples) |
| &nbsp;&nbsp;&nbsp;Notes | CUDA samples from https://github.com/NVIDIA/cuda-samples installed under /opt/cuda-samples |

| **`cuda:11.4`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `cuda` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['<36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) |
| &nbsp;&nbsp;&nbsp;Dependants | [`arrow:12.0.1`](/packages/arrow) [`arrow:14.0.1`](/packages/arrow) [`arrow:5.0.0`](/packages/arrow) [`audiocraft`](/packages/audio/audiocraft) [`auto_awq:0.2.4`](/packages/llm/auto_awq) [`auto_gptq:0.7.1`](/packages/llm/auto_gptq) [`awq:0.1.0`](/packages/llm/awq) [`bitsandbytes`](/packages/llm/bitsandbytes) [`bitsandbytes:builder`](/packages/llm/bitsandbytes) [`cuda-python:11.4`](/packages/cuda/cuda-python) [`cuda:11.4-samples`](/packages/cuda/cuda) [`cudf:21.10.02`](/packages/rapids/cudf) [`cudf:23.10.03`](/packages/rapids/cudf) [`cudnn`](/packages/cuda/cudnn) [`cuml`](/packages/rapids/cuml) [`cupy`](/packages/cuda/cupy) [`deepstream`](/packages/deepstream) [`efficientvit`](/packages/vit/efficientvit) [`exllama:0.0.14`](/packages/llm/exllama) [`exllama:0.0.15`](/packages/llm/exllama) [`faiss:1.7.3`](/packages/vectordb/faiss) [`faiss:1.7.3-builder`](/packages/vectordb/faiss) [`faiss:1.7.4`](/packages/vectordb/faiss) [`faiss:1.7.4-builder`](/packages/vectordb/faiss) [`faiss_lite`](/packages/vectordb/faiss_lite) [`flash-attention`](/packages/llm/flash-attention) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`gstreamer`](/packages/gstreamer) [`jetson-inference`](/packages/jetson-inference) [`jetson-utils`](/packages/jetson-utils) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`langchain`](/packages/llm/langchain) [`langchain:samples`](/packages/llm/langchain) [`llama_cpp:0.2.57`](/packages/llm/llama_cpp) [`llava`](/packages/llm/llava) [`local_llm`](/packages/llm/local_llm) [`minigpt4`](/packages/llm/minigpt4) [`mlc:51fb0f4`](/packages/llm/mlc) [`mlc:607dc5a`](/packages/llm/mlc) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`nemo`](/packages/nemo) [`numba`](/packages/numba) [`ollama`](/packages/llm/ollama) [`onnxruntime:1.11`](/packages/onnxruntime) [`onnxruntime:1.11-builder`](/packages/onnxruntime) [`onnxruntime:1.16.3`](/packages/onnxruntime) [`onnxruntime:1.16.3-builder`](/packages/onnxruntime) [`onnxruntime:1.17`](/packages/onnxruntime) [`onnxruntime:1.17-builder`](/packages/onnxruntime) [`openai-triton`](/packages/openai-triton) [`openai-triton:builder`](/packages/openai-triton) [`opencv:4.5.0`](/packages/opencv) [`opencv:4.5.0-builder`](/packages/opencv/opencv_builder) [`opencv:4.8.1`](/packages/opencv) [`opencv:4.8.1-builder`](/packages/opencv/opencv_builder) [`opencv:4.9.0`](/packages/opencv) [`optimum`](/packages/llm/optimum) [`piper-tts`](/packages/audio/piper-tts) [`pycuda`](/packages/cuda/pycuda) [`pytorch:1.10`](/packages/pytorch) [`pytorch:1.9`](/packages/pytorch) [`pytorch:2.0`](/packages/pytorch) [`pytorch:2.1`](/packages/pytorch) [`pytorch:2.2`](/packages/pytorch) [`pytorch:2.3`](/packages/pytorch) [`raft`](/packages/rapids/raft) [`realsense`](/packages/realsense) [`ros:foxy-desktop`](/packages/ros) [`ros:foxy-ros-base`](/packages/ros) [`ros:foxy-ros-core`](/packages/ros) [`ros:galactic-desktop`](/packages/ros) [`ros:galactic-ros-base`](/packages/ros) [`ros:galactic-ros-core`](/packages/ros) [`ros:humble-desktop`](/packages/ros) [`ros:humble-ros-base`](/packages/ros) [`ros:humble-ros-core`](/packages/ros) [`ros:iron-desktop`](/packages/ros) [`ros:iron-ros-base`](/packages/ros) [`ros:iron-ros-core`](/packages/ros) [`ros:melodic-desktop`](/packages/ros) [`ros:melodic-ros-base`](/packages/ros) [`ros:melodic-ros-core`](/packages/ros) [`ros:noetic-desktop`](/packages/ros) [`ros:noetic-ros-base`](/packages/ros) [`ros:noetic-ros-core`](/packages/ros) [`sam`](/packages/vit/sam) [`stable-diffusion`](/packages/diffusion/stable-diffusion) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`tensorflow`](/packages/tensorflow) [`tensorflow2`](/packages/tensorflow) [`tensorrt`](/packages/tensorrt) [`tensorrt_llm:0.10.dev0`](/packages/llm/tensorrt_llm) [`tensorrt_llm:0.10.dev0-builder`](/packages/llm/tensorrt_llm) [`tensorrt_llm:0.5`](/packages/llm/tensorrt_llm) [`tensorrt_llm:0.5-builder`](/packages/llm/tensorrt_llm) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`torch2trt`](/packages/pytorch/torch2trt) [`torch_tensorrt`](/packages/pytorch/torch_tensorrt) [`torchaudio:0.10.0`](/packages/pytorch/torchaudio) [`torchaudio:0.9.0`](/packages/pytorch/torchaudio) [`torchaudio:2.0.1`](/packages/pytorch/torchaudio) [`torchaudio:2.1.0`](/packages/pytorch/torchaudio) [`torchaudio:2.2.2`](/packages/pytorch/torchaudio) [`torchaudio:2.3.0`](/packages/pytorch/torchaudio) [`torchvision:0.10.0`](/packages/pytorch/torchvision) [`torchvision:0.11.1`](/packages/pytorch/torchvision) [`torchvision:0.15.1`](/packages/pytorch/torchvision) [`torchvision:0.16.2`](/packages/pytorch/torchvision) [`torchvision:0.17.2`](/packages/pytorch/torchvision) [`transformers`](/packages/llm/transformers) [`transformers:git`](/packages/llm/transformers) [`transformers:nvgpt`](/packages/llm/transformers) [`tritonserver`](/packages/tritonserver) [`tvm`](/packages/tvm) [`whisper`](/packages/audio/whisper) [`whisperx`](/packages/audio/whisperx) [`xformers`](/packages/llm/xformers) [`xtts`](/packages/audio/xtts) [`zed`](/packages/zed) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.builtin`](Dockerfile.builtin) |

| **`cuda:11.4-samples`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `cuda:samples` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['<36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:11.4`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.samples`](Dockerfile.samples) |
| &nbsp;&nbsp;&nbsp;Notes | CUDA samples from https://github.com/NVIDIA/cuda-samples installed under /opt/cuda-samples |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/cuda:12.2-r36.2.0`](https://hub.docker.com/r/dustynv/cuda/tags) | `2023-12-05` | `arm64` | `3.4GB` |
| &nbsp;&nbsp;[`dustynv/cuda:12.2-samples-r36.2.0`](https://hub.docker.com/r/dustynv/cuda/tags) | `2023-12-07` | `arm64` | `4.8GB` |

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
jetson-containers run $(autotag cuda)

# or explicitly specify one of the container images above
jetson-containers run dustynv/cuda:12.2-samples-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/cuda:12.2-samples-r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag cuda)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag cuda) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build cuda
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
