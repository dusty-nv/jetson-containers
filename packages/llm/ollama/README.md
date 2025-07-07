# ollama

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* Ollama from https://github.com/ollama/ollama with CUDA enabled (found under `/bin/ollama`)
* Thanks to [`@remy415`](https://github.com/remy415) for getting Ollama working on Jetson and contributing the Dockerfile ([PR #465](https://github.com/dusty-nv/jetson-containers/pull/465))

## Ollama Server

First, start the local Ollama server as a daemon in the background, either of these ways:

```
# models cached under jetson-containers/data
jetson-containers run --name ollama $(autotag ollama)

# models cached under your user's home directory
docker run --runtime nvidia -it --rm --network=host -v ~/ollama:/ollama -e OLLAMA_MODELS=/ollama dustynv/ollama:r36.4.0
```

You can then run the ollama [client](#ollama-client) in the same container (or a different one if desired).  The default docker run CMD of the `ollama` container is [`/start_ollama`](./start_ollama), which starts the ollama server in the background and returns control to the user. The ollama server logs are saved under your mounted `jetson-containers/data/logs` directory for monitoring them outside the containers.

Setting the `$OLLAMA_MODELS` environment variable as shown above will change where ollama downloads the models to. By default, this is under your `jetson-containers/data/models/ollama` directory which is automatically mounted by `jetson-containers run`.

## Ollama Client

Start the Ollama CLI front-end with your desired [model](https://ollama.com/library) (for example: mistral 7b)

```
# if running inside the same container as launched above
/bin/ollama run mistral

# if launching a new container for the client in another terminal
jetson-containers run $(autotag ollama) /bin/ollama run mistral
```

<img src="https://github.com/dusty-nv/jetson-containers/blob/docs/docs/images/ollama_cli.gif?raw=true" width="750px"></img>

Or you can run the client outside container by installing Ollama's binaries for arm64 (without CUDA, which only the server needs)

```
# download the latest ollama release for arm64 into /bin
sudo wget https://github.com/ollama/ollama/releases/download/$(git ls-remote --refs --sort="version:refname" --tags https://github.com/ollama/ollama | cut -d/ -f3- | sed 's/-rc.*//g' | tail -n1)/ollama-linux-arm64 -O /bin/ollama
sudo chmod +x /bin/ollama

# use the client like normal (outside container)
/bin/ollama run mistral
```

## Open WebUI

To run [Open WebUI](https://github.com/open-webui/open-webui) server for client browsers to connect to, use the `open-webui` container:

```
docker run -it --rm --network=host --add-host=host.docker.internal:host-gateway ghcr.io/open-webui/open-webui:main
```

You can then navigate your browser to `http://JETSON_IP:8080`, and create a fake account to login (these credentials are only stored locally)

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/ollama_open_webui.jpg" width="800px"></img>

## Memory Usage

| Model                                                                           |          Quantization         | Memory (MB) |
|---------------------------------------------------------------------------------|:-----------------------------:|:-----------:|
| [`TheBloke/Llama-2-7B-GGUF`](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)   |  `llama-2-7b.Q4_K_S.gguf`     |    5,268    |
| [`TheBloke/Llama-2-13B-GGUF`](https://huggingface.co/TheBloke/Llama-2-13B-GGUF) | `llama-2-13b.Q4_K_S.gguf`     |    8,609    |
| [`TheBloke/LLaMA-30b-GGUF`](https://huggingface.co/TheBloke/LLaMA-30b-GGUF)     | `llama-30b.Q4_K_S.gguf`       |    19,045   |
| [`TheBloke/Llama-2-70B-GGUF`](https://huggingface.co/TheBloke/Llama-2-70B-GGUF) | `llama-2-70b.Q4_K_S.gguf`     |    37,655   |

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`ollama:0.4.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ollama:0.4.0-r36.4.0`](https://hub.docker.com/r/dustynv/ollama/tags) `(2024-11-09, 3.3GB)` |

| **`ollama:0.5.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ollama:0.5.1-r36.4.0`](https://hub.docker.com/r/dustynv/ollama/tags) `(2024-12-12, 3.3GB)` |

| **`ollama:0.5.5`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`ollama:0.5.7`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/ollama:0.5.7-r36.4.0`](https://hub.docker.com/r/dustynv/ollama/tags) `(2025-01-30, 3.1GB)` |

| **`ollama:0.6.7`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`ollama:0.7.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`ollama:0.8.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`ollama:0.9.6`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `ollama` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dependants | [`jetson-copilot`](/packages/rag/jetson-copilot) [`langchain:samples`](/packages/rag/langchain) [`llama-index:samples`](/packages/rag/llama-index) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`ollama:0.10.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/ollama:0.4.0-r36.4.0`](https://hub.docker.com/r/dustynv/ollama/tags) | `2024-11-09` | `arm64` | `3.3GB` |
| &nbsp;&nbsp;[`dustynv/ollama:0.5.1-r36.4.0`](https://hub.docker.com/r/dustynv/ollama/tags) | `2024-12-12` | `arm64` | `3.3GB` |
| &nbsp;&nbsp;[`dustynv/ollama:0.5.7-r36.4.0`](https://hub.docker.com/r/dustynv/ollama/tags) | `2025-01-30` | `arm64` | `3.1GB` |
| &nbsp;&nbsp;[`dustynv/ollama:0.6.3-r36.4.0`](https://hub.docker.com/r/dustynv/ollama/tags) | `2025-03-31` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/ollama:0.6.8-r36.4`](https://hub.docker.com/r/dustynv/ollama/tags) | `2025-05-06` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/ollama:0.6.8-r36.4-cu126-22.04`](https://hub.docker.com/r/dustynv/ollama/tags) | `2025-05-06` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/ollama:main-r36.4.0`](https://hub.docker.com/r/dustynv/ollama/tags) | `2025-03-31` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/ollama:r35.4.1`](https://hub.docker.com/r/dustynv/ollama/tags) | `2024-06-25` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/ollama:r36.2.0`](https://hub.docker.com/r/dustynv/ollama/tags) | `2024-09-02` | `arm64` | `3.5GB` |
| &nbsp;&nbsp;[`dustynv/ollama:r36.3.0`](https://hub.docker.com/r/dustynv/ollama/tags) | `2024-09-30` | `arm64` | `3.5GB` |
| &nbsp;&nbsp;[`dustynv/ollama:r36.4.0`](https://hub.docker.com/r/dustynv/ollama/tags) | `2025-04-09` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/ollama:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/ollama/tags) | `2025-04-09` | `arm64` | `4.2GB` |
| &nbsp;&nbsp;[`dustynv/ollama:r36.4.3`](https://hub.docker.com/r/dustynv/ollama/tags) | `2025-03-11` | `arm64` | `5.0GB` |

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
jetson-containers run dustynv/ollama:0.6.8-r36.4-cu126-22.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/ollama:0.6.8-r36.4-cu126-22.04
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
