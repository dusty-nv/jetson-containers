# efficientvit

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

docs.md
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`efficientvit`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`efficientvit_jp60`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/efficientvit_jp60.yml?label=efficientvit:jp60)](https://github.com/dusty-nv/jetson-containers/actions/workflows/efficientvit_jp60.yml) [![`efficientvit_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/efficientvit_jp51.yml?label=efficientvit:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/efficientvit_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch:2.2`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`opencv`](/packages/opencv) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`tensorrt`](/packages/tensorrt) [`onnxruntime`](/packages/onnxruntime) [`jupyterlab`](/packages/jupyterlab) [`sam`](/packages/vit/sam) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/efficientvit:r35.3.1`](https://hub.docker.com/r/dustynv/efficientvit/tags) `(2024-03-07, 6.5GB)`<br>[`dustynv/efficientvit:r35.4.1`](https://hub.docker.com/r/dustynv/efficientvit/tags) `(2024-01-13, 6.5GB)`<br>[`dustynv/efficientvit:r36.2.0`](https://hub.docker.com/r/dustynv/efficientvit/tags) `(2024-01-13, 8.1GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/efficientvit:r35.3.1`](https://hub.docker.com/r/dustynv/efficientvit/tags) | `2024-03-07` | `arm64` | `6.5GB` |
| &nbsp;&nbsp;[`dustynv/efficientvit:r35.4.1`](https://hub.docker.com/r/dustynv/efficientvit/tags) | `2024-01-13` | `arm64` | `6.5GB` |
| &nbsp;&nbsp;[`dustynv/efficientvit:r36.2.0`](https://hub.docker.com/r/dustynv/efficientvit/tags) | `2024-01-13` | `arm64` | `8.1GB` |

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
jetson-containers run $(autotag efficientvit)

# or explicitly specify one of the container images above
jetson-containers run dustynv/efficientvit:r35.3.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/efficientvit:r35.3.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag efficientvit)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag efficientvit) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build efficientvit
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
