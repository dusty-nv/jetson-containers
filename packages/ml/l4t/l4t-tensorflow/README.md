# l4t-tensorflow

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`l4t-tensorflow:tf1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`pybind11`](/packages/build/pybind11) [`numpy`](/packages/numeric/numpy) [`h5py`](/packages/build/h5py) [`bazel`](/packages/build/bazel) [`protobuf:cpp`](/packages/build/protobuf/protobuf_cpp) [`tensorflow`](/packages/ml/tensorflow) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pycuda`](/packages/cuda/pycuda) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/l4t-tensorflow:tf1-r32.7.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) `(2023-12-06, 0.9GB)`<br>[`dustynv/l4t-tensorflow:tf1-r35.2.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) `(2023-12-06, 5.5GB)`<br>[`dustynv/l4t-tensorflow:tf1-r35.3.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) `(2023-08-29, 5.6GB)`<br>[`dustynv/l4t-tensorflow:tf1-r35.4.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) `(2023-10-07, 5.5GB)` |

| **`l4t-tensorflow:tf2`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`pybind11`](/packages/build/pybind11) [`numpy`](/packages/numeric/numpy) [`h5py`](/packages/build/h5py) [`bazel`](/packages/build/bazel) [`protobuf:cpp`](/packages/build/protobuf/protobuf_cpp) [`tensorflow2`](/packages/ml/tensorflow) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pycuda`](/packages/cuda/pycuda) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/l4t-tensorflow:tf2-r32.7.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) `(2023-12-06, 1.0GB)`<br>[`dustynv/l4t-tensorflow:tf2-r35.2.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) `(2023-09-07, 5.7GB)`<br>[`dustynv/l4t-tensorflow:tf2-r35.3.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) `(2023-12-06, 5.7GB)`<br>[`dustynv/l4t-tensorflow:tf2-r35.4.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) `(2023-10-07, 5.7GB)`<br>[`dustynv/l4t-tensorflow:tf2-r36.2.0`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) `(2023-12-06, 7.3GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/l4t-tensorflow:tf1-r32.7.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) | `2023-12-06` | `arm64` | `0.9GB` |
| &nbsp;&nbsp;[`dustynv/l4t-tensorflow:tf1-r35.2.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) | `2023-12-06` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/l4t-tensorflow:tf1-r35.3.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) | `2023-08-29` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/l4t-tensorflow:tf1-r35.4.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) | `2023-10-07` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/l4t-tensorflow:tf2-r32.7.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) | `2023-12-06` | `arm64` | `1.0GB` |
| &nbsp;&nbsp;[`dustynv/l4t-tensorflow:tf2-r35.2.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) | `2023-09-07` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/l4t-tensorflow:tf2-r35.3.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) | `2023-12-06` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/l4t-tensorflow:tf2-r35.4.1`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) | `2023-10-07` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/l4t-tensorflow:tf2-r36.2.0`](https://hub.docker.com/r/dustynv/l4t-tensorflow/tags) | `2023-12-06` | `arm64` | `7.3GB` |

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
jetson-containers run $(autotag l4t-tensorflow)

# or explicitly specify one of the container images above
jetson-containers run dustynv/l4t-tensorflow:tf2-r32.7.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/l4t-tensorflow:tf2-r32.7.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag l4t-tensorflow)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag l4t-tensorflow) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build l4t-tensorflow
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
