# tam

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


Track Anything from https://github.com/gaomingqi/Track-Anything

The `tam` container has a default run command to launch its web server app.

Use your web browser to access `http://HOSTNAME:12212`


<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`tam`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`tensorrt`](/packages/cuda/tensorrt) [`onnxruntime`](/packages/ml/onnxruntime) [`opengl`](/packages/multimedia/opengl) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`cuda-python`](/packages/cuda/cuda-python) [`pycuda`](/packages/cuda/pycuda) [`rust`](/packages/build/rust) [`jupyterlab`](/packages/code/jupyterlab) [`sam`](/packages/vit/sam) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/tam:r35.2.1`](https://hub.docker.com/r/dustynv/tam/tags) `(2023-12-12, 6.9GB)`<br>[`dustynv/tam:r35.3.1`](https://hub.docker.com/r/dustynv/tam/tags) `(2024-01-13, 7.0GB)`<br>[`dustynv/tam:r35.4.1`](https://hub.docker.com/r/dustynv/tam/tags) `(2024-03-07, 7.0GB)`<br>[`dustynv/tam:r36.2.0`](https://hub.docker.com/r/dustynv/tam/tags) `(2024-01-13, 8.6GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/tam:r35.2.1`](https://hub.docker.com/r/dustynv/tam/tags) | `2023-12-12` | `arm64` | `6.9GB` |
| &nbsp;&nbsp;[`dustynv/tam:r35.3.1`](https://hub.docker.com/r/dustynv/tam/tags) | `2024-01-13` | `arm64` | `7.0GB` |
| &nbsp;&nbsp;[`dustynv/tam:r35.4.1`](https://hub.docker.com/r/dustynv/tam/tags) | `2024-03-07` | `arm64` | `7.0GB` |
| &nbsp;&nbsp;[`dustynv/tam:r36.2.0`](https://hub.docker.com/r/dustynv/tam/tags) | `2024-01-13` | `arm64` | `8.6GB` |

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
jetson-containers run $(autotag tam)

# or explicitly specify one of the container images above
jetson-containers run dustynv/tam:r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/tam:r35.4.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag tam)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag tam) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build tam
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
