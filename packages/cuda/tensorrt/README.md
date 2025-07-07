# tensorrt

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`tensorrt:10.3`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `tensorrt` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==r36.*', '==cu126', 'aarch64']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dependants | [`ai-toolkit`](/packages/diffusion/ai-toolkit) [`audiocraft`](/packages/speech/audiocraft) [`clip_trt`](/packages/vit/clip_trt) [`comfyui`](/packages/diffusion/comfyui) [`deepstream`](/packages/cv/deepstream) [`diffusion_policy`](/packages/diffusion/diffusion_policy) [`dli-nano-ai`](/packages/ml/dli/dli-nano-ai) [`efficientvit`](/packages/vit/efficientvit) [`faster-whisper`](/packages/speech/faster-whisper) [`fruitnerf:1.0`](/packages/3d/nerf/fruitnerf) [`holoscan`](/packages/cv/holoscan) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:compression-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:compression-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:data-tools-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:data-tools-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`jetson-inference:foxy`](/packages/cv/jetson-inference) [`jetson-inference:galactic`](/packages/cv/jetson-inference) [`jetson-inference:humble`](/packages/cv/jetson-inference) [`jetson-inference:iron`](/packages/cv/jetson-inference) [`jetson-inference:jazzy`](/packages/cv/jetson-inference) [`jetson-inference:main`](/packages/cv/jetson-inference) [`jetson-utils:v1`](/packages/multimedia/jetson-utils) [`jetson-utils:v2`](/packages/multimedia/jetson-utils) [`jupyter_clickable_image_widget`](/packages/hw/jupyter_clickable_image_widget) [`jupyterlab:4.2.0`](/packages/code/jupyterlab) [`jupyterlab:4.2.0-myst`](/packages/code/jupyterlab) [`jupyterlab:latest`](/packages/code/jupyterlab) [`jupyterlab:latest-myst`](/packages/code/jupyterlab) [`kokoro-tts:onnx`](/packages/speech/kokoro-tts/kokoro-tts-onnx) [`l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`l4t-pytorch`](/packages/ml/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/ml/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/ml/l4t/l4t-tensorflow) [`l4t-text-generation`](/packages/ml/l4t/l4t-text-generation) [`langchain:samples`](/packages/rag/langchain) [`lerobot`](/packages/robots/lerobot) [`llama-index:samples`](/packages/rag/llama-index) [`local_llm`](/packages/llm/local_llm) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`nvidia_modelopt:0.32.0`](/packages/llm/tensorrt_optimizer/nvidia-modelopt) [`onnxruntime:1.19.2`](/packages/ml/onnxruntime) [`onnxruntime:1.20`](/packages/ml/onnxruntime) [`onnxruntime:1.20.1`](/packages/ml/onnxruntime) [`onnxruntime:1.21`](/packages/ml/onnxruntime) [`onnxruntime:1.22`](/packages/ml/onnxruntime) [`onnxruntime_genai:0.8.5`](/packages/ml/onnxruntime_genai) [`opendronemap`](/packages/robots/opendronemap) [`opendronemap:node`](/packages/robots/opendronemap) [`openpi`](/packages/robots/openpi) [`optimum`](/packages/llm/optimum) [`partpacker:0.1.0`](/packages/3d/3dobjects/partpacker) [`piper-tts`](/packages/speech/piper-tts) [`piper1-tts:1.3.0`](/packages/speech/piper1-tts) [`pytorch:2.1-all`](/packages/pytorch) [`pytorch:2.2-all`](/packages/pytorch) [`pytorch:2.3-all`](/packages/pytorch) [`pytorch:2.3.1-all`](/packages/pytorch) [`pytorch:2.4-all`](/packages/pytorch) [`pytorch:2.5-all`](/packages/pytorch) [`pytorch:2.6-all`](/packages/pytorch) [`pytorch:2.7-all`](/packages/pytorch) [`pytorch:2.8-all`](/packages/pytorch) [`ros:foxy-desktop`](/packages/robots/ros) [`ros:foxy-ros-base`](/packages/robots/ros) [`ros:foxy-ros-core`](/packages/robots/ros) [`ros:galactic-desktop`](/packages/robots/ros) [`ros:galactic-ros-base`](/packages/robots/ros) [`ros:galactic-ros-core`](/packages/robots/ros) [`ros:humble-desktop`](/packages/robots/ros) [`ros:humble-ros-base`](/packages/robots/ros) [`ros:humble-ros-core`](/packages/robots/ros) [`ros:iron-desktop`](/packages/robots/ros) [`ros:iron-ros-base`](/packages/robots/ros) [`ros:iron-ros-core`](/packages/robots/ros) [`ros:jazzy-desktop`](/packages/robots/ros) [`ros:jazzy-ros-base`](/packages/robots/ros) [`ros:jazzy-ros-core`](/packages/robots/ros) [`ros:noetic-desktop`](/packages/robots/ros) [`ros:noetic-ros-base`](/packages/robots/ros) [`ros:noetic-ros-core`](/packages/robots/ros) [`sam`](/packages/vit/sam) [`sapiens`](/packages/vit/sapiens) [`sparc3d:0.1.0`](/packages/3d/3dobjects/sparc3d) [`speaches`](/packages/speech/speaches) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`tensorflow2:2.16.1`](/packages/ml/tensorflow) [`tensorflow2:2.18.0`](/packages/ml/tensorflow) [`tensorflow2:2.19.0`](/packages/ml/tensorflow) [`tensorflow2:2.20.0`](/packages/ml/tensorflow) [`tensorflow2:2.21.0`](/packages/ml/tensorflow) [`tensorflow_graphics:2.18.0`](/packages/ml/tensorflow/graphics) [`tensorflow_graphics:2.19.0`](/packages/ml/tensorflow/graphics) [`tensorflow_graphics:2.20.0`](/packages/ml/tensorflow/graphics) [`tensorflow_text:2.18.0`](/packages/ml/tensorflow/text) [`tensorflow_text:2.19.0`](/packages/ml/tensorflow/text) [`tensorflow_text:2.20.0`](/packages/ml/tensorflow/text) [`tensorrt_llm:0.12`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.22.0`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`torch2trt`](/packages/pytorch/torch2trt) [`torch_tensorrt`](/packages/pytorch/torch_tensorrt) [`tritonserver`](/packages/ml/tritonserver) [`voice-pro`](/packages/speech/voice-pro) [`voicecraft`](/packages/speech/voicecraft) [`vscode:cuda`](/packages/code/vscode) [`vscode:torch`](/packages/code/vscode) [`warp:1.7.0-all`](/packages/numeric/warp) [`warp:1.8.1-all`](/packages/numeric/warp) [`whisper`](/packages/speech/whisper) [`whisper_trt`](/packages/speech/whisper_trt) [`whisperx`](/packages/speech/whisperx) [`wyoming-piper:1.6.2`](/packages/smart-home/wyoming/wyoming-piper) [`wyoming-piper:master`](/packages/smart-home/wyoming/wyoming-piper) [`wyoming-whisper:2.5.0`](/packages/smart-home/wyoming/wyoming-whisper) [`wyoming-whisper:master`](/packages/smart-home/wyoming/wyoming-whisper) [`xtts`](/packages/speech/xtts) [`zed:5.0`](/packages/hw/zed) [`zed:5.0-humble`](/packages/hw/zed) [`zed:5.0-jazzy`](/packages/hw/zed) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.tar`](Dockerfile.tar) |

| **`tensorrt:10.4`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==r36.*', '==cu126', 'aarch64']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.tar`](Dockerfile.tar) |

| **`tensorrt:10.5`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==r36.*', '==cu126', 'aarch64']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.tar`](Dockerfile.tar) |

| **`tensorrt:10.7`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==r36.*', '==cu126', 'aarch64']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.tar`](Dockerfile.tar) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/tensorrt:8.6-r36.2.0`](https://hub.docker.com/r/dustynv/tensorrt/tags) | `2023-12-05` | `arm64` | `6.7GB` |

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
jetson-containers run $(autotag tensorrt)

# or explicitly specify one of the container images above
jetson-containers run dustynv/tensorrt:8.6-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/tensorrt:8.6-r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag tensorrt)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag tensorrt) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build tensorrt
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
