# cudnn

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`cudnn:8.9`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `cudnn` |
| &nbsp;&nbsp;&nbsp;Builds | [![`cudnn-89_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cudnn-89_jp60.yml?label=cudnn-89:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cudnn-89_jp60.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ==36.*` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`cuda`](/packages/cuda/cuda) |
| &nbsp;&nbsp;&nbsp;Dependants | [`audiocraft`](/packages/audio/audiocraft) [`auto_gptq`](/packages/llm/auto_gptq) [`awq`](/packages/llm/awq) [`awq:dev`](/packages/llm/awq) [`bitsandbytes`](/packages/llm/bitsandbytes) [`cudnn`](/packages/cuda/cudnn) [`deepstream`](/packages/deepstream) [`efficientvit`](/packages/vit/efficientvit) [`exllama:v1`](/packages/llm/exllama) [`exllama:v2`](/packages/llm/exllama) [`faiss:lite`](/packages/vectordb/faiss_lite) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`gstreamer`](/packages/gstreamer) [`jetson-inference`](/packages/jetson-inference) [`jetson-utils`](/packages/jetson-utils) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`langchain`](/packages/llm/langchain) [`langchain:samples`](/packages/llm/langchain) [`llama_cpp:ggml`](/packages/llm/llama_cpp) [`llama_cpp:gguf`](/packages/llm/llama_cpp) [`llava`](/packages/llm/llava) [`local_llm`](/packages/llm/local_llm) [`minigpt4`](/packages/llm/minigpt4) [`mlc:51fb0f4`](/packages/llm/mlc) [`mlc:9bf5723`](/packages/llm/mlc) [`mlc:dev`](/packages/llm/mlc) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`nemo`](/packages/nemo) [`onnxruntime`](/packages/onnxruntime) [`opencv:4.5.0`](/packages/opencv) [`opencv:4.5.0-builder`](/packages/opencv/opencv_builder) [`opencv:4.8.1`](/packages/opencv) [`opencv:4.8.1-builder`](/packages/opencv/opencv_builder) [`optimum`](/packages/llm/optimum) [`pytorch:1.10`](/packages/pytorch) [`pytorch:1.11`](/packages/pytorch) [`pytorch:1.12`](/packages/pytorch) [`pytorch:1.13`](/packages/pytorch) [`pytorch:1.9`](/packages/pytorch) [`pytorch:2.0`](/packages/pytorch) [`pytorch:2.0-distributed`](/packages/pytorch) [`pytorch:2.1`](/packages/pytorch) [`pytorch:2.1-builder`](/packages/pytorch) [`pytorch:2.1-distributed`](/packages/pytorch) [`raft`](/packages/rapids/raft) [`ros:foxy-desktop`](/packages/ros) [`ros:foxy-ros-base`](/packages/ros) [`ros:foxy-ros-core`](/packages/ros) [`ros:galactic-desktop`](/packages/ros) [`ros:galactic-ros-base`](/packages/ros) [`ros:galactic-ros-core`](/packages/ros) [`ros:humble-desktop`](/packages/ros) [`ros:humble-ros-base`](/packages/ros) [`ros:humble-ros-core`](/packages/ros) [`ros:iron-desktop`](/packages/ros) [`ros:iron-ros-base`](/packages/ros) [`ros:iron-ros-core`](/packages/ros) [`ros:melodic-desktop`](/packages/ros) [`ros:melodic-ros-base`](/packages/ros) [`ros:melodic-ros-core`](/packages/ros) [`ros:noetic-desktop`](/packages/ros) [`ros:noetic-ros-base`](/packages/ros) [`ros:noetic-ros-core`](/packages/ros) [`sam`](/packages/vit/sam) [`stable-diffusion`](/packages/diffusion/stable-diffusion) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`tensorflow`](/packages/tensorflow) [`tensorflow2`](/packages/tensorflow) [`tensorrt`](/packages/tensorrt) [`tensorrt:8.6`](/packages/tensorrt) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`torch2trt`](/packages/pytorch/torch2trt) [`torch_tensorrt`](/packages/pytorch/torch_tensorrt) [`torchaudio`](/packages/pytorch/torchaudio) [`torchvision`](/packages/pytorch/torchvision) [`transformers`](/packages/llm/transformers) [`transformers:git`](/packages/llm/transformers) [`transformers:nvgpt`](/packages/llm/transformers) [`tvm`](/packages/tvm) [`whisper`](/packages/audio/whisper) [`xformers`](/packages/llm/xformers) [`zed`](/packages/zed) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/cudnn:8.9-r36.2.0`](https://hub.docker.com/r/dustynv/cudnn/tags) `(2023-12-05, 4.9GB)` |

| **`cudnn`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T <36` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`cuda`](/packages/cuda/cuda) |
| &nbsp;&nbsp;&nbsp;Dependants | [`audiocraft`](/packages/audio/audiocraft) [`auto_gptq`](/packages/llm/auto_gptq) [`awq`](/packages/llm/awq) [`awq:dev`](/packages/llm/awq) [`bitsandbytes`](/packages/llm/bitsandbytes) [`cudnn`](/packages/cuda/cudnn) [`deepstream`](/packages/deepstream) [`efficientvit`](/packages/vit/efficientvit) [`exllama:v1`](/packages/llm/exllama) [`exllama:v2`](/packages/llm/exllama) [`faiss:lite`](/packages/vectordb/faiss_lite) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`gstreamer`](/packages/gstreamer) [`jetson-inference`](/packages/jetson-inference) [`jetson-utils`](/packages/jetson-utils) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`langchain`](/packages/llm/langchain) [`langchain:samples`](/packages/llm/langchain) [`llama_cpp:ggml`](/packages/llm/llama_cpp) [`llama_cpp:gguf`](/packages/llm/llama_cpp) [`llava`](/packages/llm/llava) [`local_llm`](/packages/llm/local_llm) [`minigpt4`](/packages/llm/minigpt4) [`mlc:51fb0f4`](/packages/llm/mlc) [`mlc:9bf5723`](/packages/llm/mlc) [`mlc:dev`](/packages/llm/mlc) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`nemo`](/packages/nemo) [`onnxruntime`](/packages/onnxruntime) [`opencv:4.5.0`](/packages/opencv) [`opencv:4.5.0-builder`](/packages/opencv/opencv_builder) [`opencv:4.8.1`](/packages/opencv) [`opencv:4.8.1-builder`](/packages/opencv/opencv_builder) [`optimum`](/packages/llm/optimum) [`pytorch:1.10`](/packages/pytorch) [`pytorch:1.11`](/packages/pytorch) [`pytorch:1.12`](/packages/pytorch) [`pytorch:1.13`](/packages/pytorch) [`pytorch:1.9`](/packages/pytorch) [`pytorch:2.0`](/packages/pytorch) [`pytorch:2.0-distributed`](/packages/pytorch) [`pytorch:2.1`](/packages/pytorch) [`pytorch:2.1-builder`](/packages/pytorch) [`pytorch:2.1-distributed`](/packages/pytorch) [`raft`](/packages/rapids/raft) [`ros:foxy-desktop`](/packages/ros) [`ros:foxy-ros-base`](/packages/ros) [`ros:foxy-ros-core`](/packages/ros) [`ros:galactic-desktop`](/packages/ros) [`ros:galactic-ros-base`](/packages/ros) [`ros:galactic-ros-core`](/packages/ros) [`ros:humble-desktop`](/packages/ros) [`ros:humble-ros-base`](/packages/ros) [`ros:humble-ros-core`](/packages/ros) [`ros:iron-desktop`](/packages/ros) [`ros:iron-ros-base`](/packages/ros) [`ros:iron-ros-core`](/packages/ros) [`ros:melodic-desktop`](/packages/ros) [`ros:melodic-ros-base`](/packages/ros) [`ros:melodic-ros-core`](/packages/ros) [`ros:noetic-desktop`](/packages/ros) [`ros:noetic-ros-base`](/packages/ros) [`ros:noetic-ros-core`](/packages/ros) [`sam`](/packages/vit/sam) [`stable-diffusion`](/packages/diffusion/stable-diffusion) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`tensorflow`](/packages/tensorflow) [`tensorflow2`](/packages/tensorflow) [`tensorrt`](/packages/tensorrt) [`tensorrt:8.6`](/packages/tensorrt) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`torch2trt`](/packages/pytorch/torch2trt) [`torch_tensorrt`](/packages/pytorch/torch_tensorrt) [`torchaudio`](/packages/pytorch/torchaudio) [`torchvision`](/packages/pytorch/torchvision) [`transformers`](/packages/llm/transformers) [`transformers:git`](/packages/llm/transformers) [`transformers:nvgpt`](/packages/llm/transformers) [`tvm`](/packages/tvm) [`whisper`](/packages/audio/whisper) [`xformers`](/packages/llm/xformers) [`zed`](/packages/zed) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/cudnn:8.9-r36.2.0`](https://hub.docker.com/r/dustynv/cudnn/tags) `(2023-12-05, 4.9GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/cudnn:8.9-r36.2.0`](https://hub.docker.com/r/dustynv/cudnn/tags) | `2023-12-05` | `arm64` | `4.9GB` |

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
./run.sh $(./autotag cudnn)

# or explicitly specify one of the container images above
./run.sh dustynv/cudnn:8.9-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/cudnn:8.9-r36.2.0
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag cudnn)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag cudnn) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh cudnn
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
