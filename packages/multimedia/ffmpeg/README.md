# ffmpeg

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`ffmpeg:apt`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `ffmpeg` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) |
| &nbsp;&nbsp;&nbsp;Dependants | [`3dgrut:2.0.0`](/packages/3d/gaussian_splatting/3dgrut) [`4k4d:0.0.0`](/packages/3d/gaussian_splatting/4k4d) [`audiocraft`](/packages/speech/audiocraft) [`awq:0.1.0`](/packages/llm/awq) [`comfyui`](/packages/diffusion/comfyui) [`cosmos-predict2`](/packages/diffusion/cosmos/cosmos-predict2) [`cosmos-reason1`](/packages/diffusion/cosmos/cosmos-reason1) [`cosmos-transfer1`](/packages/diffusion/cosmos/cosmos-transfer1) [`cosmos1-diffusion-renderer:1.0.4`](/packages/diffusion/cosmos/cosmos_diffusion_renderer) [`crossformer`](/packages/vla/crossformer) [`decord2:1.0.0`](/packages/multimedia/decord) [`deepstream`](/packages/cv/deepstream) [`diffusion_policy`](/packages/diffusion/diffusion_policy) [`dli-nano-ai`](/packages/ml/dli/dli-nano-ai) [`dynamo:0.3.2`](/packages/llm/dynamo/dynamo) [`easyvolcap:0.0.0`](/packages/3d/gaussian_splatting/easyvolcap) [`efficientvit`](/packages/vit/efficientvit) [`fast_gauss:1.0.0`](/packages/3d/gaussian_splatting/fast_gauss) [`framepack`](/packages/diffusion/framepack) [`fruitnerf:1.0`](/packages/3d/nerf/fruitnerf) [`genesis-world:0.2.2`](/packages/sim/genesis) [`glomap:2.0.0`](/packages/3d/3dvision/glomap) [`gsplat:1.5.3`](/packages/3d/gaussian_splatting/gsplat) [`gstreamer`](/packages/multimedia/gstreamer) [`hloc:1.4`](/packages/3d/3dvision/hloc) [`hloc:1.5`](/packages/3d/3dvision/hloc) [`holoscan`](/packages/cv/holoscan) [`homeassistant-core:2025.7.0`](/packages/smart-home/homeassistant-core) [`isaac-gr00t`](/packages/vla/isaac-gr00t) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:data-tools-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:data-tools-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaaclab:2.2.0`](/packages/sim/isaac-sim/isaac-lab) [`isaacsim:5.0.0`](/packages/sim/isaac-sim) [`jetcam`](/packages/hw/jetcam) [`jetson-inference:foxy`](/packages/cv/jetson-inference) [`jetson-inference:galactic`](/packages/cv/jetson-inference) [`jetson-inference:humble`](/packages/cv/jetson-inference) [`jetson-inference:iron`](/packages/cv/jetson-inference) [`jetson-inference:jazzy`](/packages/cv/jetson-inference) [`jetson-inference:main`](/packages/cv/jetson-inference) [`jetson-utils:v1`](/packages/multimedia/jetson-utils) [`jetson-utils:v2`](/packages/multimedia/jetson-utils) [`kokoro-tts:fastapi`](/packages/speech/kokoro-tts/kokoro-tts-fastapi) [`l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) [`l4t-dynamo`](/packages/ml/l4t/l4t-dynamo) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`l4t-pytorch`](/packages/ml/l4t/l4t-pytorch) [`l4t-tensorflow:tf1`](/packages/ml/l4t/l4t-tensorflow) [`l4t-tensorflow:tf2`](/packages/ml/l4t/l4t-tensorflow) [`libcom:0.1.0`](/packages/multimedia/libcom) [`lita`](/packages/vlm/lita) [`llama-factory`](/packages/llm/llama-factory) [`lobechat`](/packages/llm/lobe_chat) [`local_llm`](/packages/llm/local_llm) [`memvid:0.1.4`](/packages/rag/memvid) [`mimicgen`](/packages/sim/mimicgen) [`minference:0.1.7`](/packages/llm/minference) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`nanoowl`](/packages/vit/nanoowl) [`nerfstudio:1.1.7`](/packages/3d/nerf/nerfstudio) [`nerfview:0.1.4`](/packages/3d/gaussian_splatting/nerfview) [`octo`](/packages/vla/octo) [`open3d:1.19.0`](/packages/3d/3dvision/open3d) [`opencv:4.10.0`](/packages/cv/opencv) [`opencv:4.10.0-meta`](/packages/cv/opencv) [`opencv:4.11.0`](/packages/cv/opencv) [`opencv:4.11.0-meta`](/packages/cv/opencv) [`opencv:4.12.0`](/packages/cv/opencv) [`opencv:4.12.0-meta`](/packages/cv/opencv) [`opencv:4.8.1`](/packages/cv/opencv) [`opencv:4.8.1-deb`](/packages/cv/opencv) [`opencv:4.8.1-meta`](/packages/cv/opencv) [`opendronemap`](/packages/robots/opendronemap) [`opendronemap:node`](/packages/robots/opendronemap) [`openvla:mimicgen`](/packages/vla/openvla) [`paraattention:0.4.0`](/packages/attention/ParaAttention) [`partpacker:0.1.0`](/packages/3d/3dobjects/partpacker) [`pixsfm:1.0`](/packages/3d/3dvision/pixsfm) [`protomotions:2.5.0`](/packages/robots/protomotions) [`pycolmap:3.12`](/packages/3d/3dvision/pycolmap) [`pycolmap:3.13`](/packages/3d/3dvision/pycolmap) [`robogen`](/packages/sim/robogen) [`robomimic`](/packages/sim/robomimic) [`robopoint`](/packages/vla/robopoint) [`robosuite`](/packages/sim/robosuite) [`ros:foxy-desktop`](/packages/robots/ros) [`ros:foxy-ros-base`](/packages/robots/ros) [`ros:foxy-ros-core`](/packages/robots/ros) [`ros:galactic-desktop`](/packages/robots/ros) [`ros:galactic-ros-base`](/packages/robots/ros) [`ros:galactic-ros-core`](/packages/robots/ros) [`ros:humble-desktop`](/packages/robots/ros) [`ros:humble-ros-base`](/packages/robots/ros) [`ros:humble-ros-core`](/packages/robots/ros) [`ros:iron-desktop`](/packages/robots/ros) [`ros:iron-ros-base`](/packages/robots/ros) [`ros:iron-ros-core`](/packages/robots/ros) [`ros:jazzy-desktop`](/packages/robots/ros) [`ros:jazzy-ros-base`](/packages/robots/ros) [`ros:jazzy-ros-core`](/packages/robots/ros) [`ros:noetic-desktop`](/packages/robots/ros) [`ros:noetic-ros-base`](/packages/robots/ros) [`ros:noetic-ros-core`](/packages/robots/ros) [`sage-attention:3.0.0`](/packages/attention/sage-attention) [`sam`](/packages/vit/sam) [`sapiens`](/packages/vit/sapiens) [`sdnext`](/packages/diffusion/sdnext) [`self-forcing`](/packages/diffusion/self-forcing) [`sglang:0.4.4`](/packages/llm/sglang) [`sglang:0.4.6`](/packages/llm/sglang) [`sglang:0.4.9`](/packages/llm/sglang) [`sparc3d:0.1.0`](/packages/3d/3dobjects/sparc3d) [`sparge-attention:0.1.0`](/packages/attention/sparge-attention) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`sudonim:hf`](/packages/llm/sudonim) [`tam`](/packages/vit/tam) [`video-codec-sdk:12.2.72-samples`](/packages/multimedia/video-codec-sdk) [`video-codec-sdk:13.0.19-samples`](/packages/multimedia/video-codec-sdk) [`videollama:1.0.0`](/packages/vlm/videollama) [`vila`](/packages/vlm/vila) [`vllm:0.7.4`](/packages/llm/vllm) [`vllm:0.8.4`](/packages/llm/vllm) [`vllm:0.9.0`](/packages/llm/vllm) [`vllm:0.9.2`](/packages/llm/vllm) [`vllm:0.9.3`](/packages/llm/vllm) [`vllm:v0.8.5.post1`](/packages/llm/vllm) [`voice-pro`](/packages/speech/voice-pro) [`voicecraft`](/packages/speech/voicecraft) [`xattention:0.0.1`](/packages/attention/xattention) [`zed:5.0-humble`](/packages/hw/zed) [`zed:5.0-jazzy`](/packages/hw/zed) [`zigma:1.0`](/packages/ml/mamba/zigma) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/FFmpeg/FFmpeg |

| **`ffmpeg:7.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `ffmpeg:git` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`cuda`](/packages/cuda/cuda) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) |
| &nbsp;&nbsp;&nbsp;Dependants | [`cv-cuda:0.15`](/packages/cv/cv-cuda) [`isaac-ros:compression-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:compression-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`lerobot`](/packages/robots/lerobot) [`openpi`](/packages/robots/openpi) [`pyav`](/packages/multimedia/pyav) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/FFmpeg/FFmpeg |

| **`ffmpeg:jetpack`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases |  |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['aarch64']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/FFmpeg/FFmpeg |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag ffmpeg)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host ffmpeg:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag ffmpeg)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag ffmpeg) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build ffmpeg
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
