# onnxruntime

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`onnxruntime:1.22`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `onnxruntime` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36', '>=cu126']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`onnx`](/packages/ml/onnx) |
| &nbsp;&nbsp;&nbsp;Dependants | [`ai-toolkit`](/packages/diffusion/ai-toolkit) [`comfyui`](/packages/diffusion/comfyui) [`efficientvit`](/packages/vit/efficientvit) [`faster-whisper`](/packages/speech/faster-whisper) [`fruitnerf:1.0`](/packages/3d/nerf/fruitnerf) [`kokoro-tts:onnx`](/packages/speech/kokoro-tts/kokoro-tts-onnx) [`l4t-diffusion`](/packages/ml/l4t/l4t-diffusion) [`l4t-ml`](/packages/ml/l4t/l4t-ml) [`local_llm`](/packages/llm/local_llm) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) [`onnxruntime_genai:0.8.5`](/packages/ml/onnxruntime_genai) [`opendronemap`](/packages/robots/opendronemap) [`opendronemap:node`](/packages/robots/opendronemap) [`optimum`](/packages/llm/optimum) [`partpacker:0.1.0`](/packages/3d/3dobjects/partpacker) [`piper-tts`](/packages/speech/piper-tts) [`piper1-tts:1.3.0`](/packages/speech/piper1-tts) [`sam`](/packages/vit/sam) [`sparc3d:0.1.0`](/packages/3d/3dobjects/sparc3d) [`speaches`](/packages/speech/speaches) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`voice-pro`](/packages/speech/voice-pro) [`voicecraft`](/packages/speech/voicecraft) [`whisper_trt`](/packages/speech/whisper_trt) [`whisperx`](/packages/speech/whisperx) [`wyoming-piper:1.6.2`](/packages/smart-home/wyoming/wyoming-piper) [`wyoming-piper:master`](/packages/smart-home/wyoming/wyoming-piper) [`wyoming-whisper:2.5.0`](/packages/smart-home/wyoming/wyoming-whisper) [`wyoming-whisper:master`](/packages/smart-home/wyoming/wyoming-whisper) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/onnxruntime:1.22-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/onnxruntime/tags) `(2025-03-03, 5.2GB)` |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/microsoft/onnxruntime |

| **`onnxruntime:1.21`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36', '>=cu124']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`onnx`](/packages/ml/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/microsoft/onnxruntime |

| **`onnxruntime:1.20.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36', '>=cu124']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`onnx`](/packages/ml/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/microsoft/onnxruntime |

| **`onnxruntime:1.20`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36', '>=cu124']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`onnx`](/packages/ml/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/onnxruntime:1.20-r36.4.0`](https://hub.docker.com/r/dustynv/onnxruntime/tags) `(2024-10-13, 5.7GB)` |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/microsoft/onnxruntime |

| **`onnxruntime:1.19.2`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36', '>=cu124']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`onnx`](/packages/ml/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/microsoft/onnxruntime |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/onnxruntime:1.20-r36.4.0`](https://hub.docker.com/r/dustynv/onnxruntime/tags) | `2024-10-13` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/onnxruntime:1.20.2-r36.4.0`](https://hub.docker.com/r/dustynv/onnxruntime/tags) | `2025-02-19` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/onnxruntime:1.22-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/onnxruntime/tags) | `2025-03-03` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/onnxruntime:r32.7.1`](https://hub.docker.com/r/dustynv/onnxruntime/tags) | `2023-12-11` | `arm64` | `0.5GB` |
| &nbsp;&nbsp;[`dustynv/onnxruntime:r35.2.1`](https://hub.docker.com/r/dustynv/onnxruntime/tags) | `2023-12-12` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/onnxruntime:r35.3.1`](https://hub.docker.com/r/dustynv/onnxruntime/tags) | `2023-11-13` | `arm64` | `5.2GB` |
| &nbsp;&nbsp;[`dustynv/onnxruntime:r35.4.1`](https://hub.docker.com/r/dustynv/onnxruntime/tags) | `2023-11-08` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/onnxruntime:r36.2.0`](https://hub.docker.com/r/dustynv/onnxruntime/tags) | `2023-12-12` | `arm64` | `6.9GB` |

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
jetson-containers run $(autotag onnxruntime)

# or explicitly specify one of the container images above
jetson-containers run dustynv/onnxruntime:1.22-r36.4.0-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/onnxruntime:1.22-r36.4.0-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag onnxruntime)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag onnxruntime) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build onnxruntime
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
