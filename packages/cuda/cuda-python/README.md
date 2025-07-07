# cuda-python

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`cuda-python:12.2`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu122`](/packages/cuda/cuda) [`cuda:12.2`](/packages/cuda/cuda) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`cuda-python:12.4`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu124`](/packages/cuda/cuda) [`cuda:12.4`](/packages/cuda/cuda) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`cuda-python:12.6`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `cuda-python` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) |
| &nbsp;&nbsp;&nbsp;Dependants | [`3dgrut:2.0.0`](/packages/3d/gaussian_splatting/3dgrut) [`audiocraft`](/packages/speech/audiocraft) [`awq:0.1.0`](/packages/llm/awq) [`block_sparse_attn:0.0.1`](/packages/attention/block-sparse-attention) [`comfyui`](/packages/diffusion/comfyui) [`cosmos-predict2`](/packages/diffusion/cosmos/cosmos-predict2) [`cosmos-reason1`](/packages/diffusion/cosmos/cosmos-reason1) [`cosmos-transfer1`](/packages/diffusion/cosmos/cosmos-transfer1) [`cosmos1-diffusion-renderer:1.0.4`](/packages/diffusion/cosmos/cosmos_diffusion_renderer) [`cudf:23.10.03`](/packages/ml/rapids/cudf) [`cudf:25.08.00`](/packages/ml/rapids/cudf) [`cuml`](/packages/ml/rapids/cuml) [`cutlass:4.1.0`](/packages/cuda/cutlass) [`diffusion_policy`](/packages/diffusion/diffusion_policy) [`dli-nano-ai`](/packages/ml/dli/dli-nano-ai) [`dynamo:0.3.2`](/packages/llm/dynamo/dynamo) [`efficientvit`](/packages/vit/efficientvit) [`exllama:0.1`](/packages/llm/exllama) [`faiss_lite`](/packages/vectordb/faiss_lite) [`flash-attention:2.5.7`](/packages/attention/flash-attention) [`flash-attention:2.6.3`](/packages/attention/flash-attention) [`flash-attention:2.7.2.post1`](/packages/attention/flash-attention) [`flash-attention:2.7.4.post1`](/packages/attention/flash-attention) [`flash-attention:2.8.0.post2`](/packages/attention/flash-attention) [`flash-attention:2.8.1`](/packages/attention/flash-attention) [`flexprefill:0.1.0`](/packages/attention/flexprefill) [`framepack`](/packages/diffusion/framepack) [`genai-bench:0.1.0`](/packages/llm/sglang/genai-bench) [`huggingface_kernels:0.7.0`](/packages/attention/huggingface_kernels) [`hymba`](/packages/llm/hymba) [`isaac-gr00t`](/packages/vla/isaac-gr00t) [`isaac-ros:compression-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:compression-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`jupyter_clickable_image_widget`](/packages/hw/jupyter_clickable_image_widget) [`jupyterlab:4.2.0`](/packages/code/jupyterlab) [`jupyterlab:4.2.0-myst`](/packages/code/jupyterlab) [`jupyterlab:latest`](/packages/code/jupyterlab) [`jupyterlab:latest-myst`](/packages/code/jupyterlab) [`l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) [`l4t-dynamo`](/packages/ml/l4t/l4t-dynamo) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`l4t-text-generation`](/packages/ml/l4t/l4t-text-generation) [`langchain:samples`](/packages/rag/langchain) [`lerobot`](/packages/robots/lerobot) [`lita`](/packages/vlm/lita) [`llama-factory`](/packages/llm/llama-factory) [`llama-index:samples`](/packages/rag/llama-index) [`llava`](/packages/vlm/llava) [`lobechat`](/packages/llm/lobe_chat) [`local_llm`](/packages/llm/local_llm) [`minference:0.1.7`](/packages/llm/minference) [`mlstm_kernels:2.0.1`](/packages/ml/xlstm/mlstm_kernels) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`nanodb`](/packages/vectordb/nanodb) [`nvidia_modelopt:0.32.0`](/packages/llm/tensorrt_optimizer/nvidia-modelopt) [`openpi`](/packages/robots/openpi) [`openvla`](/packages/vla/openvla) [`openvla:mimicgen`](/packages/vla/openvla) [`paraattention:0.4.0`](/packages/attention/ParaAttention) [`partpacker:0.1.0`](/packages/3d/3dobjects/partpacker) [`plstm:0.1.0`](/packages/ml/xlstm/pltsm) [`prismatic`](/packages/vlm/prismatic) [`pytorch:2.1-all`](/packages/pytorch) [`pytorch:2.2-all`](/packages/pytorch) [`pytorch:2.3-all`](/packages/pytorch) [`pytorch:2.3.1-all`](/packages/pytorch) [`pytorch:2.4-all`](/packages/pytorch) [`pytorch:2.5-all`](/packages/pytorch) [`pytorch:2.6-all`](/packages/pytorch) [`pytorch:2.7-all`](/packages/pytorch) [`pytorch:2.8-all`](/packages/pytorch) [`raft`](/packages/ml/rapids/raft) [`robopoint`](/packages/vla/robopoint) [`sage-attention:3.0.0`](/packages/attention/sage-attention) [`sam`](/packages/vit/sam) [`sdnext`](/packages/diffusion/sdnext) [`self-forcing`](/packages/diffusion/self-forcing) [`sgl-kernel:0.2.3`](/packages/llm/sglang/sgl-kernel) [`sglang:0.4.4`](/packages/llm/sglang) [`sglang:0.4.6`](/packages/llm/sglang) [`sglang:0.4.9`](/packages/llm/sglang) [`sparc3d:0.1.0`](/packages/3d/3dobjects/sparc3d) [`sparge-attention:0.1.0`](/packages/attention/sparge-attention) [`sudonim:hf`](/packages/llm/sudonim) [`tam`](/packages/vit/tam) [`tensorrt_llm:0.12`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.22.0`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`transformer-engine:2.7`](/packages/ml/transformer-engine) [`videollama:1.0.0`](/packages/vlm/videollama) [`vila`](/packages/vlm/vila) [`vllm:0.7.4`](/packages/llm/vllm) [`vllm:0.8.4`](/packages/llm/vllm) [`vllm:0.9.0`](/packages/llm/vllm) [`vllm:0.9.2`](/packages/llm/vllm) [`vllm:0.9.3`](/packages/llm/vllm) [`vllm:v0.8.5.post1`](/packages/llm/vllm) [`voice-pro`](/packages/speech/voice-pro) [`voicecraft`](/packages/speech/voicecraft) [`vscode:transformers`](/packages/code/vscode) [`warp:1.7.0-all`](/packages/numeric/warp) [`warp:1.8.1-all`](/packages/numeric/warp) [`whisper`](/packages/speech/whisper) [`xattention:0.0.1`](/packages/attention/xattention) [`xlstm:2.0.5`](/packages/ml/xlstm/xlstm) [`xtuner`](/packages/vlm/xtuner) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`cuda-python:12.8`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu128`](/packages/cuda/cuda) [`cuda:12.8`](/packages/cuda/cuda) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`cuda-python:12.9`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu129`](/packages/cuda/cuda) [`cuda:12.9`](/packages/cuda/cuda) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`cuda-python:13.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu130`](/packages/cuda/cuda) [`cuda:13.0`](/packages/cuda/cuda) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/cuda-python:builder-r35.4.1`](https://hub.docker.com/r/dustynv/cuda-python/tags) | `2024-03-26` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/cuda-python:builder-r36.2.0`](https://hub.docker.com/r/dustynv/cuda-python/tags) | `2024-03-11` | `arm64` | `3.6GB` |
| &nbsp;&nbsp;[`dustynv/cuda-python:r35.2.1`](https://hub.docker.com/r/dustynv/cuda-python/tags) | `2023-12-06` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/cuda-python:r35.3.1`](https://hub.docker.com/r/dustynv/cuda-python/tags) | `2023-08-29` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/cuda-python:r35.4.1`](https://hub.docker.com/r/dustynv/cuda-python/tags) | `2023-12-06` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/cuda-python:r36.2.0`](https://hub.docker.com/r/dustynv/cuda-python/tags) | `2023-12-06` | `arm64` | `3.5GB` |
| &nbsp;&nbsp;[`dustynv/cuda-python:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/cuda-python/tags) | `2025-03-03` | `arm64` | `2.3GB` |

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
jetson-containers run $(autotag cuda-python)

# or explicitly specify one of the container images above
jetson-containers run dustynv/cuda-python:r36.4.0-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/cuda-python:r36.4.0-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag cuda-python)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag cuda-python) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build cuda-python
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
