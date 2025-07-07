# comfyui

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

docs.md
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`comfyui`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35.0.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`ffmpeg`](/packages/multimedia/ffmpeg) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) [`torchsde`](/packages/pytorch/torchsde) [`pytorch3d`](/packages/pytorch/torch3d) [`triton`](/packages/ml/triton) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`transformer-engine`](/packages/ml/transformer-engine) [`torch-memory-saver`](/packages/pytorch/torchsaver) [`opengl`](/packages/multimedia/opengl) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`opencv`](/packages/cv/opencv) [`bitsandbytes`](/packages/llm/bitsandbytes) [`torchao`](/packages/pytorch/torchao) [`sage-attention`](/packages/attention/sage-attention) [`sparge-attention`](/packages/attention/sparge-attention) [`flexprefill`](/packages/attention/flexprefill) [`paraattention`](/packages/attention/ParaAttention) [`tensorrt`](/packages/cuda/tensorrt) [`onnxruntime`](/packages/ml/onnxruntime) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/comfyui:r36.3.0`](https://hub.docker.com/r/dustynv/comfyui/tags) `(2024-08-24, 5.9GB)`<br>[`dustynv/comfyui:r36.4.0`](https://hub.docker.com/r/dustynv/comfyui/tags) `(2024-12-30, 8.4GB)`<br>[`dustynv/comfyui:r36.4.3`](https://hub.docker.com/r/dustynv/comfyui/tags) `(2025-03-11, 5.9GB)`<br>[`dustynv/comfyui:r36.4.3-cu128-24.04`](https://hub.docker.com/r/dustynv/comfyui/tags) `(2025-03-11, 5.2GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/comfyui:r36.3.0`](https://hub.docker.com/r/dustynv/comfyui/tags) | `2024-08-24` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/comfyui:r36.4.0`](https://hub.docker.com/r/dustynv/comfyui/tags) | `2024-12-30` | `arm64` | `8.4GB` |
| &nbsp;&nbsp;[`dustynv/comfyui:r36.4.3`](https://hub.docker.com/r/dustynv/comfyui/tags) | `2025-03-11` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/comfyui:r36.4.3-cu128-24.04`](https://hub.docker.com/r/dustynv/comfyui/tags) | `2025-03-11` | `arm64` | `5.2GB` |

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
jetson-containers run $(autotag comfyui)

# or explicitly specify one of the container images above
jetson-containers run dustynv/comfyui:r36.4.3

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/comfyui:r36.4.3
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag comfyui)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag comfyui) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build comfyui
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
