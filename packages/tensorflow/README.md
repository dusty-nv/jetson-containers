# tensorflow

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

Container for TF1/TF2 with CUDA support.
Note that the [`l4t-tensorflow`](/packages/l4t/l4t-tensorflow) containers are similar, with the addition of OpenCV and PyCUDA.  

The TensorFlow wheels used in these are from https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`tensorflow`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`tensorflow_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tensorflow_jp46.yml?label=tensorflow:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tensorflow_jp46.yml) [![`tensorflow_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tensorflow_jp51.yml?label=tensorflow:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tensorflow_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/tensorflow:r32.7.1`](https://hub.docker.com/r/dustynv/tensorflow/tags) `(2023-09-07, 0.8GB)`<br>[`dustynv/tensorflow:r35.2.1`](https://hub.docker.com/r/dustynv/tensorflow/tags) `(2023-09-07, 5.5GB)`<br>[`dustynv/tensorflow:r35.3.1`](https://hub.docker.com/r/dustynv/tensorflow/tags) `(2023-08-29, 5.5GB)`<br>[`dustynv/tensorflow:r35.4.1`](https://hub.docker.com/r/dustynv/tensorflow/tags) `(2023-10-07, 5.5GB)` |

| **`tensorflow2`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`tensorflow2_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tensorflow2_jp51.yml?label=tensorflow2:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tensorflow2_jp51.yml) [![`tensorflow2_jp46`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/tensorflow2_jp46.yml?label=tensorflow2:jp46)](https://github.com/dusty-nv/jetson-containers/actions/workflows/tensorflow2_jp46.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) |
| &nbsp;&nbsp;&nbsp;Dependants | [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/tensorflow2:r32.7.1`](https://hub.docker.com/r/dustynv/tensorflow2/tags) `(2023-09-07, 0.9GB)`<br>[`dustynv/tensorflow2:r35.2.1`](https://hub.docker.com/r/dustynv/tensorflow2/tags) `(2023-09-07, 5.6GB)`<br>[`dustynv/tensorflow2:r35.3.1`](https://hub.docker.com/r/dustynv/tensorflow2/tags) `(2023-08-29, 5.6GB)`<br>[`dustynv/tensorflow2:r35.4.1`](https://hub.docker.com/r/dustynv/tensorflow2/tags) `(2023-10-07, 5.6GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/tensorflow:r32.7.1`](https://hub.docker.com/r/dustynv/tensorflow/tags) | `2023-09-07` | `arm64` | `0.8GB` |
| &nbsp;&nbsp;[`dustynv/tensorflow:r35.2.1`](https://hub.docker.com/r/dustynv/tensorflow/tags) | `2023-09-07` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/tensorflow:r35.3.1`](https://hub.docker.com/r/dustynv/tensorflow/tags) | `2023-08-29` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/tensorflow:r35.4.1`](https://hub.docker.com/r/dustynv/tensorflow/tags) | `2023-10-07` | `arm64` | `5.5GB` |

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
./run.sh $(./autotag tensorflow)

# or explicitly specify one of the container images above
./run.sh dustynv/tensorflow:r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/tensorflow:r35.4.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag tensorflow)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag tensorflow) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh tensorflow
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
