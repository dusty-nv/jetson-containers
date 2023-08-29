# l4t-pytorch

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`l4t-pytorch`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`l4t-pytorch_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp46.yml?label=l4t-pytorch:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-pytorch_jp46.yml) [![`l4t-pytorch_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/l4t-pytorch_jp51.yml?label=l4t-pytorch:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/l4t-pytorch_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) [`torch2trt`](/packages/pytorch/torch2trt) [`opencv`](/packages/opencv) [`pycuda`](/packages/pycuda) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/l4t-pytorch:r32.7.1`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) `(2023-08-12, 1.2GB)`<br>[`dustynv/l4t-pytorch:r35.2.1`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) `(2023-08-12, 5.5GB)`<br>[`dustynv/l4t-pytorch:r35.3.1`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) `(2023-08-13, 5.6GB)`<br>[`dustynv/l4t-pytorch:r35.4.1`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) `(2023-08-13, 5.5GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/l4t-pytorch:r32.7.1`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) | `2023-08-12` | `arm64` | `1.2GB` |
| &nbsp;&nbsp;[`dustynv/l4t-pytorch:r35.2.1`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) | `2023-08-12` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/l4t-pytorch:r35.3.1`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) | `2023-08-13` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/l4t-pytorch:r35.4.1`](https://hub.docker.com/r/dustynv/l4t-pytorch/tags) | `2023-08-13` | `arm64` | `5.5GB` |

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
./run.sh $(./autotag l4t-pytorch)

# or explicitly specify one of the container images above
./run.sh dustynv/l4t-pytorch:r35.3.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/l4t-pytorch:r35.3.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag l4t-pytorch)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag l4t-pytorch) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh l4t-pytorch
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
