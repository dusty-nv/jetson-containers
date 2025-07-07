# l4t-text-generation

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`l4t-text-generation`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`llvm:17`](/packages/build/llvm) [`sudonim`](/packages/llm/sudonim) [`flashinfer:0.2.8`](/packages/attention/flash-infer) [`mlc`](/packages/llm/mlc) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`cuda-python`](/packages/cuda/cuda-python) [`cutlass`](/packages/cuda/cutlass) [`flash-attention`](/packages/attention/flash-attention) [`exllama`](/packages/llm/exllama) [`llama_cpp`](/packages/llm/llama_cpp) [`tensorrt`](/packages/cuda/tensorrt) [`pycuda`](/packages/cuda/pycuda) [`jupyterlab`](/packages/code/jupyterlab) [`docker`](/packages/build/docker) [`openai`](/packages/llm/openai) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/l4t-text-generation:r35.2.1`](https://hub.docker.com/r/dustynv/l4t-text-generation/tags) `(2023-12-12, 6.4GB)`<br>[`dustynv/l4t-text-generation:r35.3.1`](https://hub.docker.com/r/dustynv/l4t-text-generation/tags) `(2024-02-22, 6.7GB)`<br>[`dustynv/l4t-text-generation:r35.4.1`](https://hub.docker.com/r/dustynv/l4t-text-generation/tags) `(2023-12-19, 6.4GB)`<br>[`dustynv/l4t-text-generation:r36.2.0`](https://hub.docker.com/r/dustynv/l4t-text-generation/tags) `(2024-03-12, 10.4GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/l4t-text-generation:r35.2.1`](https://hub.docker.com/r/dustynv/l4t-text-generation/tags) | `2023-12-12` | `arm64` | `6.4GB` |
| &nbsp;&nbsp;[`dustynv/l4t-text-generation:r35.3.1`](https://hub.docker.com/r/dustynv/l4t-text-generation/tags) | `2024-02-22` | `arm64` | `6.7GB` |
| &nbsp;&nbsp;[`dustynv/l4t-text-generation:r35.4.1`](https://hub.docker.com/r/dustynv/l4t-text-generation/tags) | `2023-12-19` | `arm64` | `6.4GB` |
| &nbsp;&nbsp;[`dustynv/l4t-text-generation:r36.2.0`](https://hub.docker.com/r/dustynv/l4t-text-generation/tags) | `2024-03-12` | `arm64` | `10.4GB` |

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
jetson-containers run $(autotag l4t-text-generation)

# or explicitly specify one of the container images above
jetson-containers run dustynv/l4t-text-generation:r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/l4t-text-generation:r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag l4t-text-generation)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag l4t-text-generation) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build l4t-text-generation
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
