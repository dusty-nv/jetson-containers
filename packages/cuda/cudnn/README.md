# cudnn

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`cudnn:8.9`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`cudnn-89_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/cudnn-89_jp60.yml?label=cudnn-89:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/cudnn-89_jp60.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==36.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) |
| &nbsp;&nbsp;&nbsp;Dependants | [`tensorrt:8.6`](/packages/tensorrt) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/cudnn:8.9-r36.2.0`](https://hub.docker.com/r/dustynv/cudnn/tags) `(2023-12-05, 4.9GB)` |

| **`cudnn:9.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==36.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.4`](/packages/cuda/cuda) |
| &nbsp;&nbsp;&nbsp;Dependants | [`tensorrt:10.0`](/packages/tensorrt) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`cudnn`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['<36']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) |
| &nbsp;&nbsp;&nbsp;Dependants | [`audiocraft`](/packages/audio/audiocraft) [`auto_awq:0.2.4`](/packages/llm/auto_awq) [`auto_gptq:0.7.1`](/packages/llm/auto_gptq) [`awq:0.1.0`](/packages/llm/awq) [`bitsandbytes`](/packages/llm/bitsandbytes) [`bitsandbytes:builder`](/packages/llm/bitsandbytes) [`ctranslate2:4.2.0`](/packages/ctranslate2) [`ctranslate2:4.2.0-builder`](/packages/ctranslate2) [`ctranslate2:master`](/packages/ctranslate2) [`ctranslate2:master-builder`](/packages/ctranslate2) [`deepstream`](/packages/deepstream) [`efficientvit`](/packages/vit/efficientvit) [`exllama:0.0.14`](/packages/llm/exllama) [`exllama:0.0.15`](/packages/llm/exllama) [`faiss_lite`](/packages/vectordb/faiss_lite) [`faster-whisper`](/packages/audio/faster-whisper) [`flash-attention:2.5.6`](/packages/llm/flash-attention) [`flash-attention:2.5.6-builder`](/packages/llm/flash-attention) [`flash-attention:2.5.7`](/packages/llm/flash-attention) [`flash-attention:2.5.7-builder`](/packages/llm/flash-attention) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`gstreamer`](/packages/gstreamer) [`jetson-inference`](/packages/jetson-inference) [`jetson-utils`](/packages/jetson-utils) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) [`langchain`](/packages/rag/langchain) [`langchain:samples`](/packages/rag/langchain) [`llama-index`](/packages/rag/llama-index) [`llama_cpp:0.2.57`](/packages/llm/llama_cpp) [`llava`](/packages/llm/llava) [`minigpt4`](/packages/llm/minigpt4) [`mlc:0.1.0`](/packages/llm/mlc) [`mlc:0.1.0-builder`](/packages/llm/mlc) [`mlc:0.1.1`](/packages/llm/mlc) [`mlc:0.1.1-builder`](/packages/llm/mlc) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`nemo`](/packages/nemo) [`onnxruntime:1.11`](/packages/onnxruntime) [`onnxruntime:1.11-builder`](/packages/onnxruntime) [`onnxruntime:1.16.3`](/packages/onnxruntime) [`onnxruntime:1.16.3-builder`](/packages/onnxruntime) [`onnxruntime:1.17`](/packages/onnxruntime) [`onnxruntime:1.17-builder`](/packages/onnxruntime) [`onnxruntime:1.19`](/packages/onnxruntime) [`onnxruntime:1.19-builder`](/packages/onnxruntime) [`openai-triton`](/packages/openai-triton) [`openai-triton:builder`](/packages/openai-triton) [`opencv:4.5.0`](/packages/opencv) [`opencv:4.5.0-builder`](/packages/opencv) [`opencv:4.8.1`](/packages/opencv) [`opencv:4.8.1-builder`](/packages/opencv) [`opencv:4.9.0`](/packages/opencv) [`opencv:4.9.0-builder`](/packages/opencv) [`optimum`](/packages/llm/optimum) [`piper-tts`](/packages/audio/piper-tts) [`pytorch:1.10`](/packages/pytorch) [`pytorch:1.9`](/packages/pytorch) [`pytorch:2.0`](/packages/pytorch) [`pytorch:2.0-builder`](/packages/pytorch) [`pytorch:2.1`](/packages/pytorch) [`pytorch:2.1-builder`](/packages/pytorch) [`pytorch:2.2`](/packages/pytorch) [`pytorch:2.2-builder`](/packages/pytorch) [`pytorch:2.3`](/packages/pytorch) [`pytorch:2.3-builder`](/packages/pytorch) [`raft`](/packages/rapids/raft) [`ros:foxy-desktop`](/packages/ros) [`ros:foxy-ros-base`](/packages/ros) [`ros:foxy-ros-core`](/packages/ros) [`ros:galactic-desktop`](/packages/ros) [`ros:galactic-ros-base`](/packages/ros) [`ros:galactic-ros-core`](/packages/ros) [`ros:humble-desktop`](/packages/ros) [`ros:humble-ros-base`](/packages/ros) [`ros:humble-ros-core`](/packages/ros) [`ros:iron-desktop`](/packages/ros) [`ros:iron-ros-base`](/packages/ros) [`ros:iron-ros-core`](/packages/ros) [`ros:melodic-desktop`](/packages/ros) [`ros:melodic-ros-base`](/packages/ros) [`ros:melodic-ros-core`](/packages/ros) [`ros:noetic-desktop`](/packages/ros) [`ros:noetic-ros-base`](/packages/ros) [`ros:noetic-ros-core`](/packages/ros) [`sam`](/packages/vit/sam) [`stable-diffusion`](/packages/diffusion/stable-diffusion) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`tensorflow`](/packages/tensorflow) [`tensorflow2`](/packages/tensorflow) [`tensorrt`](/packages/tensorrt) [`tensorrt_llm:0.10.dev0`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.10.dev0-builder`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.5`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.5-builder`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`torch2trt`](/packages/pytorch/torch2trt) [`torch_tensorrt`](/packages/pytorch/torch_tensorrt) [`torchaudio:0.10.0`](/packages/pytorch/torchaudio) [`torchaudio:0.10.0-builder`](/packages/pytorch/torchaudio) [`torchaudio:0.9.0`](/packages/pytorch/torchaudio) [`torchaudio:0.9.0-builder`](/packages/pytorch/torchaudio) [`torchaudio:2.0.1`](/packages/pytorch/torchaudio) [`torchaudio:2.0.1-builder`](/packages/pytorch/torchaudio) [`torchaudio:2.1.0`](/packages/pytorch/torchaudio) [`torchaudio:2.1.0-builder`](/packages/pytorch/torchaudio) [`torchaudio:2.2.2`](/packages/pytorch/torchaudio) [`torchaudio:2.2.2-builder`](/packages/pytorch/torchaudio) [`torchaudio:2.3.0`](/packages/pytorch/torchaudio) [`torchaudio:2.3.0-builder`](/packages/pytorch/torchaudio) [`torchvision:0.10.0`](/packages/pytorch/torchvision) [`torchvision:0.11.1`](/packages/pytorch/torchvision) [`torchvision:0.15.1`](/packages/pytorch/torchvision) [`torchvision:0.16.2`](/packages/pytorch/torchvision) [`torchvision:0.17.2`](/packages/pytorch/torchvision) [`torchvision:0.18.0`](/packages/pytorch/torchvision) [`transformers`](/packages/llm/transformers) [`transformers:git`](/packages/llm/transformers) [`transformers:nvgpt`](/packages/llm/transformers) [`tritonserver`](/packages/tritonserver) [`tvm`](/packages/tvm) [`voicecraft`](/packages/audio/voicecraft) [`whisper`](/packages/audio/whisper) [`whisperx`](/packages/audio/whisperx) [`wyoming-piper:master`](/packages/smart-home/wyoming/piper) [`wyoming-whisper:latest`](/packages/smart-home/wyoming/wyoming-whisper) [`xformers:0.0.26`](/packages/llm/xformers) [`xformers:0.0.26-builder`](/packages/llm/xformers) [`xtts`](/packages/audio/xtts) [`zed`](/packages/hardware/zed) |
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

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag cudnn)

# or explicitly specify one of the container images above
jetson-containers run dustynv/cudnn:8.9-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/cudnn:8.9-r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag cudnn)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag cudnn) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build cudnn
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
