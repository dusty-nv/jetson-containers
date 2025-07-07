# tensorrt_llm

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`tensorrt_llm:0.12`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['<cu128']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`cuda-python`](/packages/cuda/cuda-python) [`ninja`](/packages/build/ninja) [`gdrcopy`](/packages/cuda/gdrcopy) [`torchaudio`](/packages/pytorch/torchaudio) [`triton`](/packages/ml/triton) [`torchao`](/packages/pytorch/torchao) [`mooncake`](/packages/llm/dynamo/mooncake) [`nixl`](/packages/llm/dynamo/nixl) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/tensorrt_llm:0.12-r36.4.0`](https://hub.docker.com/r/dustynv/tensorrt_llm/tags) `(2024-11-13, 8.2GB)` |
| &nbsp;&nbsp;&nbsp;Notes | The `tensorrt-llm:builder` container includes the C++ binaries under `/opt` |

| **`tensorrt_llm:0.22.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `tensorrt_llm` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=cu126']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`cuda-python`](/packages/cuda/cuda-python) [`ninja`](/packages/build/ninja) [`gdrcopy`](/packages/cuda/gdrcopy) [`torchaudio`](/packages/pytorch/torchaudio) [`triton`](/packages/ml/triton) [`torchao`](/packages/pytorch/torchao) [`mooncake`](/packages/llm/dynamo/mooncake) [`nixl`](/packages/llm/dynamo/nixl) |
| &nbsp;&nbsp;&nbsp;Dependants | [`nvidia_modelopt:0.32.0`](/packages/llm/tensorrt_optimizer/nvidia-modelopt) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The `tensorrt-llm:builder` container includes the C++ binaries under `/opt` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/tensorrt_llm:0.12-r36.4.0`](https://hub.docker.com/r/dustynv/tensorrt_llm/tags) | `2024-11-13` | `arm64` | `8.2GB` |

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
jetson-containers run $(autotag tensorrt_llm)

# or explicitly specify one of the container images above
jetson-containers run dustynv/tensorrt_llm:0.12-r36.4.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/tensorrt_llm:0.12-r36.4.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag tensorrt_llm)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag tensorrt_llm) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build tensorrt_llm
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
