# jetson-inference

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`jetson-inference`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`cuda`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch:2.2`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`tensorrt`](/packages/tensorrt) [`opencv`](/packages/opencv) [`gstreamer`](/packages/gstreamer) |
| &nbsp;&nbsp;&nbsp;Dependants | [`local_llm`](/packages/llm/local_llm) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/jetson-inference:22.06`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2022-09-30, 6.5GB)`<br>[`dustynv/jetson-inference:r32.4.3`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2020-10-27, 0.9GB)`<br>[`dustynv/jetson-inference:r32.4.4`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2021-11-16, 0.9GB)`<br>[`dustynv/jetson-inference:r32.5.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2021-08-09, 0.9GB)`<br>[`dustynv/jetson-inference:r32.6.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2021-08-24, 0.9GB)`<br>[`dustynv/jetson-inference:r32.7.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2023-05-15, 1.1GB)`<br>[`dustynv/jetson-inference:r34.1.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2022-04-08, 5.9GB)`<br>[`dustynv/jetson-inference:r34.1.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2023-03-18, 6.1GB)`<br>[`dustynv/jetson-inference:r35.1.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2023-05-15, 6.1GB)`<br>[`dustynv/jetson-inference:r35.2.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2023-05-15, 6.0GB)`<br>[`dustynv/jetson-inference:r35.3.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2023-05-15, 5.6GB)`<br>[`dustynv/jetson-inference:r35.4.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2023-08-30, 5.7GB)`<br>[`dustynv/jetson-inference:r36.2.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) `(2023-12-19, 7.9GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/jetson-inference:22.06`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2022-09-30` | `amd64` | `6.5GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r32.4.3`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2020-10-27` | `arm64` | `0.9GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r32.4.4`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2021-11-16` | `arm64` | `0.9GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r32.5.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2021-08-09` | `arm64` | `0.9GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r32.6.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2021-08-24` | `arm64` | `0.9GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r32.7.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2023-05-15` | `arm64` | `1.1GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r34.1.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2022-04-08` | `arm64` | `5.9GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r34.1.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2023-03-18` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r35.1.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2023-05-15` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r35.2.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2023-05-15` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r35.3.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2023-05-15` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r35.4.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2023-08-30` | `arm64` | `5.7GB` |
| &nbsp;&nbsp;[`dustynv/jetson-inference:r36.2.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | `2023-12-19` | `arm64` | `7.9GB` |

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
jetson-containers run $(autotag jetson-inference)

# or explicitly specify one of the container images above
jetson-containers run dustynv/jetson-inference:r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/jetson-inference:r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag jetson-inference)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag jetson-inference) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build jetson-inference
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
