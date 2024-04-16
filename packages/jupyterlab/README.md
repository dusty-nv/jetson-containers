# jupyterlab

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`jupyterlab`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`jupyterlab_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jupyterlab_jp46.yml?label=jupyterlab:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jupyterlab_jp46.yml) [![`jupyterlab_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jupyterlab_jp51.yml?label=jupyterlab:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jupyterlab_jp51.yml) [![`jupyterlab_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/jupyterlab_jp60.yml?label=jupyterlab:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/jupyterlab_jp60.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`rust`](/packages/build/rust) |
| &nbsp;&nbsp;&nbsp;Dependants | [`audiocraft`](/packages/audio/audiocraft) [`efficientvit`](/packages/vit/efficientvit) [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-text-generation`](/packages/l4t/l4t-text-generation) [`langchain:samples`](/packages/llm/langchain) [`sam`](/packages/vit/sam) [`tam`](/packages/vit/tam) [`whisper`](/packages/audio/whisper) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/jupyterlab:r32.7.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) `(2024-03-07, 0.7GB)`<br>[`dustynv/jupyterlab:r35.2.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) `(2023-12-06, 5.3GB)`<br>[`dustynv/jupyterlab:r35.3.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) `(2024-03-07, 5.4GB)`<br>[`dustynv/jupyterlab:r35.4.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) `(2023-10-07, 5.3GB)`<br>[`dustynv/jupyterlab:r36.2.0`](https://hub.docker.com/r/dustynv/jupyterlab/tags) `(2024-03-07, 0.6GB)` |
| &nbsp;&nbsp;&nbsp;Notes | will autostart Jupyter server on port 8888 unless container entry CMD is overridden |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r32.7.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2024-03-07` | `arm64` | `0.7GB` |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r35.2.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2023-12-06` | `arm64` | `5.3GB` |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r35.3.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2024-03-07` | `arm64` | `5.4GB` |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r35.4.1`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2023-10-07` | `arm64` | `5.3GB` |
| &nbsp;&nbsp;[`dustynv/jupyterlab:r36.2.0`](https://hub.docker.com/r/dustynv/jupyterlab/tags) | `2024-03-07` | `arm64` | `0.6GB` |

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
jetson-containers run $(autotag jupyterlab)

# or explicitly specify one of the container images above
jetson-containers run dustynv/jupyterlab:r35.3.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/jupyterlab:r35.3.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag jupyterlab)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag jupyterlab) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build jupyterlab
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
