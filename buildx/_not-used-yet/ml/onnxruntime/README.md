# onnxruntime

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`onnxruntime:1.19`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `onnxruntime` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36', '>=cu124']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numpy) [`onnx`](/packages/onnx) |
| &nbsp;&nbsp;&nbsp;Dependants | [`efficientvit`](/packages/vit/efficientvit) [`l4t-diffusion`](/packages/l4t/l4t-diffusion) [`l4t-ml`](/packages/l4t/l4t-ml) [`optimum`](/packages/llm/optimum) [`piper-tts`](/packages/audio/piper-tts) [`sam`](/packages/vit/sam) [`stable-diffusion-webui`](/packages/diffusion/stable-diffusion-webui) [`tam`](/packages/vit/tam) [`wyoming-piper:master`](/packages/smart-home/wyoming/piper) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | the `onnxruntime-gpu` wheel that's built is saved in the container under `/opt` |

| **`onnxruntime:1.19-builder`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `onnxruntime:builder` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36', '>=cu124']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numpy) [`onnx`](/packages/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | the `onnxruntime-gpu` wheel that's built is saved in the container under `/opt` |

| **`onnxruntime:1.17`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `onnxruntime` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36', '<=cu122']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numpy) [`onnx`](/packages/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | the `onnxruntime-gpu` wheel that's built is saved in the container under `/opt` |

| **`onnxruntime:1.17-builder`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `onnxruntime:builder` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36', '<=cu122']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numpy) [`onnx`](/packages/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | the `onnxruntime-gpu` wheel that's built is saved in the container under `/opt` |

| **`onnxruntime:1.16.3`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `onnxruntime` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==35.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numpy) [`onnx`](/packages/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | the `onnxruntime-gpu` wheel that's built is saved in the container under `/opt` |

| **`onnxruntime:1.16.3-builder`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `onnxruntime:builder` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==35.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numpy) [`onnx`](/packages/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | the `onnxruntime-gpu` wheel that's built is saved in the container under `/opt` |

| **`onnxruntime:1.11`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `onnxruntime` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==32.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numpy) [`onnx`](/packages/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | the `onnxruntime-gpu` wheel that's built is saved in the container under `/opt` |

| **`onnxruntime:1.11-builder`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `onnxruntime:builder` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['==32.*']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numpy) [`onnx`](/packages/onnx) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | the `onnxruntime-gpu` wheel that's built is saved in the container under `/opt` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
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
jetson-containers run dustynv/onnxruntime:r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/onnxruntime:r36.2.0
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
