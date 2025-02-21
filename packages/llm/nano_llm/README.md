# nano_llm

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


> [!NOTE]  
> [`NanoLLM`](https://dusty-nv.github.io/NanoLLM) is a lightweight, optimized library for LLM inference and multimodal agents.
> For more info, see these resources:
> * Repo - [`github.com/dusty-nv/NanoLLM`](https://github.com/dusty-nv/NanoLLM)
> * Docs - [`dusty-nv.github.io/NanoLLM`](https://dusty-nv.github.io/NanoLLM)
> * Jetson AI Lab - [Live Llava](https://www.jetson-ai-lab.com/tutorial_live-llava.html), [NanoVLM](https://www.jetson-ai-lab.com/tutorial_nano-vlm.html), [SLM](https://www.jetson-ai-lab.com/tutorial_slm.html)
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`nano_llm:main`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `nano_llm` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=35']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch:2.2`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`awq`](/packages/llm/awq) [`mlc`](/packages/llm/mlc) [`riva-client:python`](/packages/audio/riva-client) [`tensorrt`](/packages/tensorrt) [`opencv`](/packages/opencv) [`gstreamer`](/packages/gstreamer) [`jetson-inference`](/packages/jetson-inference) [`torch2trt`](/packages/pytorch/torch2trt) [`torchaudio`](/packages/pytorch/torchaudio) [`onnxruntime`](/packages/onnxruntime) [`piper-tts`](/packages/audio/piper-tts) [`whisper_trt`](/packages/audio/whisper_trt) [`cuda-python`](/packages/cuda/cuda-python) [`faiss`](/packages/vectordb/faiss) [`faiss_lite`](/packages/vectordb/faiss_lite) [`clip_trt`](/packages/vit/clip_trt) [`nanodb`](/packages/vectordb/nanodb) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/nano_llm:24.4-r35.4.1`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-04-15` | `arm64` | `8.5GB` |
| &nbsp;&nbsp;[`dustynv/nano_llm:24.4-r36.2.0`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-04-15` | `arm64` | `9.7GB` |
| &nbsp;&nbsp;[`dustynv/nano_llm:24.4.1-r35.4.1`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-04-19` | `arm64` | `10.2GB` |
| &nbsp;&nbsp;[`dustynv/nano_llm:24.4.1-r36.2.0`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-04-19` | `arm64` | `11.1GB` |
| &nbsp;&nbsp;[`dustynv/nano_llm:24.5-r35.4.1`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-05-03` | `arm64` | `10.2GB` |
| &nbsp;&nbsp;[`dustynv/nano_llm:24.5-r36.2.0`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-05-03` | `arm64` | `11.2GB` |
| &nbsp;&nbsp;[`dustynv/nano_llm:24.5.1-r35.4.1`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-05-19` | `arm64` | `10.0GB` |
| &nbsp;&nbsp;[`dustynv/nano_llm:24.5.1-r36.2.0`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-05-19` | `arm64` | `10.6GB` |
| &nbsp;&nbsp;[`dustynv/nano_llm:24.6-r36.2.0`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-06-09` | `arm64` | `10.8GB` |
| &nbsp;&nbsp;[`dustynv/nano_llm:r35.4.1`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-05-15` | `arm64` | `10.0GB` |
| &nbsp;&nbsp;[`dustynv/nano_llm:r36.2.0`](https://hub.docker.com/r/dustynv/nano_llm/tags) | `2024-06-29` | `arm64` | `10.8GB` |

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
jetson-containers run $(autotag nano_llm)

# or explicitly specify one of the container images above
jetson-containers run dustynv/nano_llm:r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/nano_llm:r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag nano_llm)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag nano_llm) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build nano_llm
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
