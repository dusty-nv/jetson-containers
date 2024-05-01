# onnx

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`onnx`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`onnx_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/onnx_jp46.yml?label=onnx:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/onnx_jp46.yml) [![`onnx_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/onnx_jp51.yml?label=onnx:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/onnx_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numpy) |
| &nbsp;&nbsp;&nbsp;Dependants | [`audiocraft`](/packages/audio/audiocraft) [`auto_awq:0.2.4`](/packages/llm/auto_awq) [`auto_gptq:0.7.1`](/packages/llm/auto_gptq) [`awq:0.1.0`](/packages/llm/awq) [`bitsandbytes`](/packages/llm/bitsandbytes) [`bitsandbytes:builder`](/packages/llm/bitsandbytes) [`efficientvit`](/packages/vit/efficientvit) [`exllama:0.0.14`](/packages/llm/exllama) [`exllama:0.0.15`](/packages/llm/exllama) [`faiss_lite`](/packages/vectordb/faiss_lite) [`flash-attention`](/packages/llm/flash-attention) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`jetson-inference`](/packages/jetson-inference) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-pytorch`](/packages/l4t/l4t-pytorch) [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`langchain`](/packages/llm/langchain) [`langchain:samples`](/packages/llm/langchain) [`llava`](/packages/llm/llava) [`local_llm`](/packages/llm/local_llm) [`minigpt4`](/packages/llm/minigpt4) [`mlc:51fb0f4`](/packages/llm/mlc) [`mlc:607dc5a`](/packages/llm/mlc) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`nemo`](/packages/nemo) [`onnxruntime:1.11`](/packages/onnxruntime) [`onnxruntime:1.11-builder`](/packages/onnxruntime) [`onnxruntime:1.16.3`](/packages/onnxruntime) [`onnxruntime:1.16.3-builder`](/packages/onnxruntime) [`onnxruntime:1.17`](/packages/onnxruntime) [`onnxruntime:1.17-builder`](/packages/onnxruntime) [`openai-triton`](/packages/openai-triton) [`openai-triton:builder`](/packages/openai-triton) [`optimum`](/packages/llm/optimum) [`piper-tts`](/packages/audio/piper-tts) [`pytorch:1.10`](/packages/pytorch) [`pytorch:1.9`](/packages/pytorch) [`pytorch:2.0`](/packages/pytorch) [`pytorch:2.1`](/packages/pytorch) [`pytorch:2.2`](/packages/pytorch) [`pytorch:2.3`](/packages/pytorch) [`raft`](/packages/rapids/raft) [`sam`](/packages/vit/sam) [`stable-diffusion`](/packages/diffusion/stable-diffusion) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`tensorrt_llm:0.10.dev0`](/packages/llm/tensorrt_llm) [`tensorrt_llm:0.10.dev0-builder`](/packages/llm/tensorrt_llm) [`tensorrt_llm:0.5`](/packages/llm/tensorrt_llm) [`tensorrt_llm:0.5-builder`](/packages/llm/tensorrt_llm) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`torch2trt`](/packages/pytorch/torch2trt) [`torch_tensorrt`](/packages/pytorch/torch_tensorrt) [`torchaudio:0.10.0`](/packages/pytorch/torchaudio) [`torchaudio:0.9.0`](/packages/pytorch/torchaudio) [`torchaudio:2.0.1`](/packages/pytorch/torchaudio) [`torchaudio:2.1.0`](/packages/pytorch/torchaudio) [`torchaudio:2.2.2`](/packages/pytorch/torchaudio) [`torchaudio:2.3.0`](/packages/pytorch/torchaudio) [`torchvision:0.10.0`](/packages/pytorch/torchvision) [`torchvision:0.11.1`](/packages/pytorch/torchvision) [`torchvision:0.15.1`](/packages/pytorch/torchvision) [`torchvision:0.16.2`](/packages/pytorch/torchvision) [`torchvision:0.17.2`](/packages/pytorch/torchvision) [`transformers`](/packages/llm/transformers) [`transformers:git`](/packages/llm/transformers) [`transformers:nvgpt`](/packages/llm/transformers) [`tvm`](/packages/tvm) [`whisper`](/packages/audio/whisper) [`whisperx`](/packages/audio/whisperx) [`xformers`](/packages/llm/xformers) [`xtts`](/packages/audio/xtts) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/onnx:r32.7.1`](https://hub.docker.com/r/dustynv/onnx/tags) `(2023-12-11, 0.4GB)`<br>[`dustynv/onnx:r35.2.1`](https://hub.docker.com/r/dustynv/onnx/tags) `(2023-12-12, 5.0GB)`<br>[`dustynv/onnx:r35.3.1`](https://hub.docker.com/r/dustynv/onnx/tags) `(2023-12-11, 5.0GB)`<br>[`dustynv/onnx:r35.4.1`](https://hub.docker.com/r/dustynv/onnx/tags) `(2023-10-07, 5.0GB)` |
| &nbsp;&nbsp;&nbsp;Notes | the `protobuf_apt` is added as a dependency on JetPack 4 (newer versions of onnx build it in-tree) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/onnx:r32.7.1`](https://hub.docker.com/r/dustynv/onnx/tags) | `2023-12-11` | `arm64` | `0.4GB` |
| &nbsp;&nbsp;[`dustynv/onnx:r35.2.1`](https://hub.docker.com/r/dustynv/onnx/tags) | `2023-12-12` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/onnx:r35.3.1`](https://hub.docker.com/r/dustynv/onnx/tags) | `2023-12-11` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/onnx:r35.4.1`](https://hub.docker.com/r/dustynv/onnx/tags) | `2023-10-07` | `arm64` | `5.0GB` |

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
jetson-containers run $(autotag onnx)

# or explicitly specify one of the container images above
jetson-containers run dustynv/onnx:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/onnx:r35.2.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag onnx)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag onnx) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build onnx
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
