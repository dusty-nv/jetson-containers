# ollama

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* ollama from https://github.com/ollama/ollama with CUDA enabled (found under `/bin/ollama`)

# Container Usage

Run the container as a daemon in the background
`docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama dusty-nv/ollama`

Start the Ollama front-end with your desired model (for example: mistral 7b)
`docker run -it --rm dusty-nv/ollama /bin/ollama run mistral`

### Memory Usage

| Model                                                                           |          Quantization         | Memory (MB) |
|---------------------------------------------------------------------------------|:-----------------------------:|:-----------:|
| [`TheBloke/Llama-2-7B-GGUF`](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)   |  `llama-2-7b.Q4_K_S.gguf`     |    5,268    |
| [`TheBloke/Llama-2-13B-GGUF`](https://huggingface.co/TheBloke/Llama-2-13B-GGUF) | `llama-2-13b.Q4_K_S.gguf`     |    8,609    |
| [`TheBloke/LLaMA-30b-GGUF`](https://huggingface.co/TheBloke/LLaMA-30b-GGUF)     | `llama-30b.Q4_K_S.gguf`       |    19,045   |
| [`TheBloke/Llama-2-70B-GGUF`](https://huggingface.co/TheBloke/Llama-2-70B-GGUF) | `llama-2-70b.Q4_K_S.gguf`     |    37,655   |

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`ollama`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ollama:r35.4.1`](https://hub.docker.com/r/dustynv/ollama/tags) `(2024-04-05, 5.4GB)`<br>[`dustynv/ollama:r36.2.0`](https://hub.docker.com/r/dustynv/ollama/tags) `(2024-04-05, 3.9GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/ollama:r35.4.1`](https://hub.docker.com/r/dustynv/ollama/tags) | `2024-04-05` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/ollama:r36.2.0`](https://hub.docker.com/r/dustynv/ollama/tags) | `2024-04-05` | `arm64` | `3.9GB` |

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
jetson-containers run $(autotag ollama)

# or explicitly specify one of the container images above
jetson-containers run dustynv/ollama:r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/ollama:r35.4.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag ollama)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag ollama) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build ollama
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
