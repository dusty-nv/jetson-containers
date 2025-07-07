# optimum

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`optimum`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numeric/numpy) [`onnx`](/packages/ml/onnx) [`onnxruntime`](/packages/ml/onnxruntime) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/optimum:r32.7.1`](https://hub.docker.com/r/dustynv/optimum/tags) `(2023-12-15, 1.7GB)`<br>[`dustynv/optimum:r35.2.1`](https://hub.docker.com/r/dustynv/optimum/tags) `(2023-12-15, 6.1GB)`<br>[`dustynv/optimum:r35.3.1`](https://hub.docker.com/r/dustynv/optimum/tags) `(2023-12-14, 6.1GB)`<br>[`dustynv/optimum:r35.4.1`](https://hub.docker.com/r/dustynv/optimum/tags) `(2023-11-05, 6.1GB)` |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/huggingface/optimum |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/optimum:r32.7.1`](https://hub.docker.com/r/dustynv/optimum/tags) | `2023-12-15` | `arm64` | `1.7GB` |
| &nbsp;&nbsp;[`dustynv/optimum:r35.2.1`](https://hub.docker.com/r/dustynv/optimum/tags) | `2023-12-15` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/optimum:r35.3.1`](https://hub.docker.com/r/dustynv/optimum/tags) | `2023-12-14` | `arm64` | `6.1GB` |
| &nbsp;&nbsp;[`dustynv/optimum:r35.4.1`](https://hub.docker.com/r/dustynv/optimum/tags) | `2023-11-05` | `arm64` | `6.1GB` |

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
jetson-containers run $(autotag optimum)

# or explicitly specify one of the container images above
jetson-containers run dustynv/optimum:r35.2.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/optimum:r35.2.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag optimum)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag optimum) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build optimum
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
