# huggingface_hub

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`huggingface_hub`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`huggingface_hub_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/huggingface_hub_jp46.yml?label=huggingface_hub:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/huggingface_hub_jp46.yml) [![`huggingface_hub_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/huggingface_hub_jp51.yml?label=huggingface_hub:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/huggingface_hub_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) |
| &nbsp;&nbsp;&nbsp;Dependants | [`audiocraft`](/packages/audio/audiocraft) [`auto_awq`](/packages/llm/auto_awq) [`auto_gptq`](/packages/llm/auto_gptq) [`awq`](/packages/llm/awq) [`awq:dev`](/packages/llm/awq) [`bitsandbytes`](/packages/llm/bitsandbytes) [`efficientvit`](/packages/vit/efficientvit) [`exllama:v1`](/packages/llm/exllama) [`exllama:v2`](/packages/llm/exllama) [`faster-whisper`](/packages/audio/faster-whisper) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`langchain`](/packages/llm/langchain) [`langchain:samples`](/packages/llm/langchain) [`llama_cpp:ggml`](/packages/llm/llama_cpp) [`llama_cpp:gguf`](/packages/llm/llama_cpp) [`llava`](/packages/llm/llava) [`local_llm`](/packages/llm/local_llm) [`minigpt4`](/packages/llm/minigpt4) [`mlc:1f70d71`](/packages/llm/mlc) [`mlc:1f70d71-builder`](/packages/llm/mlc) [`mlc:3feed05`](/packages/llm/mlc) [`mlc:3feed05-builder`](/packages/llm/mlc) [`mlc:51fb0f4`](/packages/llm/mlc) [`mlc:51fb0f4-builder`](/packages/llm/mlc) [`mlc:5584cac`](/packages/llm/mlc) [`mlc:5584cac-builder`](/packages/llm/mlc) [`mlc:607dc5a`](/packages/llm/mlc) [`mlc:607dc5a-builder`](/packages/llm/mlc) [`mlc:731616e`](/packages/llm/mlc) [`mlc:731616e-builder`](/packages/llm/mlc) [`mlc:9bf5723`](/packages/llm/mlc) [`mlc:9bf5723-builder`](/packages/llm/mlc) [`mlc:dev`](/packages/llm/mlc) [`mlc:dev-builder`](/packages/llm/mlc) [`nanodb`](/packages/vectordb/nanodb) [`nanoowl`](/packages/vit/nanoowl) [`nanosam`](/packages/vit/nanosam) [`nemo`](/packages/nemo) [`optimum`](/packages/llm/optimum) [`stable-diffusion`](/packages/diffusion/stable-diffusion) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`text-generation-inference`](/packages/llm/text-generation-inference) [`text-generation-webui:1.7`](/packages/llm/text-generation-webui) [`text-generation-webui:6a7cd01`](/packages/llm/text-generation-webui) [`text-generation-webui:main`](/packages/llm/text-generation-webui) [`transformers`](/packages/llm/transformers) [`transformers:git`](/packages/llm/transformers) [`transformers:nvgpt`](/packages/llm/transformers) [`whisperx`](/packages/audio/whisperx) |
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

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag huggingface_hub)

# or explicitly specify one of the container images above
./run.sh dustynv/huggingface_hub:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/huggingface_hub:r35.2.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag huggingface_hub)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag huggingface_hub) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh huggingface_hub
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
