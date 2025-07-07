# langchain

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* langchain from https://github.com/langchain-ai/langchain

The `langchain:samples` container has a default run command to launch Jupyter Lab with notebook directory to be `/opt/langchain`

Use your web browser to access `http://HOSTNAME:8888`
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`langchain`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `langchain:main` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) [`llama_cpp`](/packages/llm/llama_cpp) |
| &nbsp;&nbsp;&nbsp;Dependants | [`langchain:samples`](/packages/rag/langchain) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/langchain:r35.2.1`](https://hub.docker.com/r/dustynv/langchain/tags) `(2023-12-06, 5.6GB)`<br>[`dustynv/langchain:r35.3.1`](https://hub.docker.com/r/dustynv/langchain/tags) `(2023-12-19, 5.6GB)`<br>[`dustynv/langchain:r35.4.1`](https://hub.docker.com/r/dustynv/langchain/tags) `(2024-01-24, 5.6GB)`<br>[`dustynv/langchain:r36.2.0`](https://hub.docker.com/r/dustynv/langchain/tags) `(2024-01-24, 7.3GB)`<br>[`dustynv/langchain:samples-r35.2.1`](https://hub.docker.com/r/dustynv/langchain/tags) `(2024-01-24, 6.0GB)`<br>[`dustynv/langchain:samples-r35.3.1`](https://hub.docker.com/r/dustynv/langchain/tags) `(2024-01-24, 6.0GB)`<br>[`dustynv/langchain:samples-r35.4.1`](https://hub.docker.com/r/dustynv/langchain/tags) `(2024-03-07, 6.2GB)`<br>[`dustynv/langchain:samples-r36.2.0`](https://hub.docker.com/r/dustynv/langchain/tags) `(2024-03-07, 7.8GB)` |

| **`langchain:samples`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) [`huggingface_hub`](/packages/llm/huggingface_hub) [`sudonim`](/packages/llm/sudonim) [`llama_cpp`](/packages/llm/llama_cpp) [`langchain:main`](/packages/rag/langchain) [`tensorrt`](/packages/cuda/tensorrt) [`cuda-python`](/packages/cuda/cuda-python) [`pycuda`](/packages/cuda/pycuda) [`rust`](/packages/build/rust) [`jupyterlab`](/packages/code/jupyterlab) [`ollama`](/packages/llm/ollama) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.samples`](Dockerfile.samples) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/langchain:samples-r35.2.1`](https://hub.docker.com/r/dustynv/langchain/tags) `(2024-01-24, 6.0GB)`<br>[`dustynv/langchain:samples-r35.3.1`](https://hub.docker.com/r/dustynv/langchain/tags) `(2024-01-24, 6.0GB)`<br>[`dustynv/langchain:samples-r35.4.1`](https://hub.docker.com/r/dustynv/langchain/tags) `(2024-03-07, 6.2GB)`<br>[`dustynv/langchain:samples-r36.2.0`](https://hub.docker.com/r/dustynv/langchain/tags) `(2024-03-07, 7.8GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/langchain:r35.2.1`](https://hub.docker.com/r/dustynv/langchain/tags) | `2023-12-06` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/langchain:r35.3.1`](https://hub.docker.com/r/dustynv/langchain/tags) | `2023-12-19` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/langchain:r35.4.1`](https://hub.docker.com/r/dustynv/langchain/tags) | `2024-01-24` | `arm64` | `5.6GB` |
| &nbsp;&nbsp;[`dustynv/langchain:r36.2.0`](https://hub.docker.com/r/dustynv/langchain/tags) | `2024-01-24` | `arm64` | `7.3GB` |
| &nbsp;&nbsp;[`dustynv/langchain:samples-r35.2.1`](https://hub.docker.com/r/dustynv/langchain/tags) | `2024-01-24` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/langchain:samples-r35.3.1`](https://hub.docker.com/r/dustynv/langchain/tags) | `2024-01-24` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/langchain:samples-r35.4.1`](https://hub.docker.com/r/dustynv/langchain/tags) | `2024-03-07` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/langchain:samples-r36.2.0`](https://hub.docker.com/r/dustynv/langchain/tags) | `2024-03-07` | `arm64` | `7.8GB` |

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
jetson-containers run $(autotag langchain)

# or explicitly specify one of the container images above
jetson-containers run dustynv/langchain:samples-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/langchain:samples-r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag langchain)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag langchain) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build langchain
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
