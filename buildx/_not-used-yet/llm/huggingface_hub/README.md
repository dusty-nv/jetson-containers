# huggingface_hub

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`huggingface_hub`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`huggingface_hub_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/huggingface_hub_jp46.yml?label=huggingface_hub:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/huggingface_hub_jp46.yml) [![`huggingface_hub_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/huggingface_hub_jp51.yml?label=huggingface_hub:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/huggingface_hub_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dependants | [`audiocraft`](/packages/audio/audiocraft) [`auto_awq:0.2.4`](/packages/llm/auto_awq) [`auto_gptq:0.7.1`](/packages/llm/auto_gptq) [`awq:0.1.0`](/packages/llm/awq) [`bitsandbytes`](/packages/llm/bitsandbytes) [`bitsandbytes:builder`](/packages/llm/bitsandbytes) [`efficientvit`](/packages/vit/efficientvit) [`exllama:0.0.14`](/packages/llm/exllama) [`exllama:0.0.15`](/packages/llm/exllama) [`faster-whisper`](/packages/audio/faster-whisper) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`langchain`](/packages/rag/langchain) [`langchain:samples`](/packages/rag/langchain) [`llama_cpp:0.2.57`](/packages/llm/llama_cpp) [`llava`](/packages/llm/llava) [`minigpt4`](/packages/llm/minigpt4) [`mlc:0.1.0`](/packages/llm/mlc) [`mlc:0.1.0-builder`](/packages/llm/mlc) [`mlc:0.1.1`](/packages/llm/mlc) [`mlc:0.1.1-builder`](/packages/llm/mlc) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`nemo`](/packages/nemo) [`optimum`](/packages/llm/optimum) [`stable-diffusion`](/packages/diffusion/stable-diffusion) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tensorrt_llm:0.10.dev0`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.10.dev0-builder`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.5`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`tensorrt_llm:0.5-builder`](/packages/llm/tensorrt_optimizer/tensorrt_llm) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`transformers`](/packages/llm/transformers) [`transformers:git`](/packages/llm/transformers) [`transformers:nvgpt`](/packages/llm/transformers) [`voicecraft`](/packages/audio/voicecraft) [`whisperx`](/packages/audio/whisperx) [`wyoming-whisper:latest`](/packages/smart-home/wyoming/wyoming-whisper) [`xtts`](/packages/audio/xtts) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/huggingface_hub:r32.7.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) `(2023-12-15, 0.4GB)`<br>[`dustynv/huggingface_hub:r35.2.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) `(2023-12-15, 5.0GB)`<br>[`dustynv/huggingface_hub:r35.3.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) `(2023-09-07, 5.0GB)`<br>[`dustynv/huggingface_hub:r35.4.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) `(2023-10-07, 5.0GB)` |
| &nbsp;&nbsp;&nbsp;Notes | provides `huggingface-cli` and `huggingface-downloader` tools |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/huggingface_hub:r32.7.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) | `2023-12-15` | `arm64` | `0.4GB` |
| &nbsp;&nbsp;[`dustynv/huggingface_hub:r35.2.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) | `2023-12-15` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/huggingface_hub:r35.3.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) | `2023-09-07` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/huggingface_hub:r35.4.1`](https://hub.docker.com/r/dustynv/huggingface_hub/tags) | `2023-10-07` | `arm64` | `5.0GB` |

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
jetson-containers run $(autotag huggingface_hub)

# or explicitly specify one of the container images above
jetson-containers run dustynv/huggingface_hub:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/huggingface_hub:r35.2.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag huggingface_hub)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag huggingface_hub) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build huggingface_hub
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
