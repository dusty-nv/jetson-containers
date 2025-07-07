# llama-index

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* llama-index from https://www.llamaindex.ai/

## Starting `llamaindex` container

```bash
jetson-containers run $(autotag llama-index:samples)
```

This will start the `ollama` server as well as Jupyter Lab server inside the container.

## Running a RAG example with Ollama

This is based on the [official tutorial for local models](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/).

#### Jupyter Notebook Version

When you run start the `llama-index` container, you should see lines like this on the terminal.

```
JupyterLab URL:   http://192.168.1.10:8888 (password "nvidia")
JupyterLab logs:  /data/logs/jupyter.log
```

On your Jetson desktop GUI, or on a PC on the same network as Jetson, open your web browser and access the address. When prompted, type the password `nvidia` and log in.

Jupyter Lab UI should show up, with [`LlamaIndex_Local-Models.ipynb`](samples/LlamaIndex_Local-Models.ipynb) listed in the left navigator pane - open it, and follow the guide in the Jupyter notebook.

####  Python Version

After starting the `llamaindex` container, you should be on `root@<hostname>` console. First, download the Llama2 model using `ollama`

```bash
ollama pull llama2
```

This downloads the default 7-billion parameter Llama2 model - you can optionally specify `ollma2:13b` and `ollma2:70b` for other variations, and change the Python script (line 13) accordingly. Then type the following to start the sample Python script:

```bash
python3 /opt/llama-index/llamaindex_starter.py
```


<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`llama-index`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `llama-index:main` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) |
| &nbsp;&nbsp;&nbsp;Dependants | [`llama-index:samples`](/packages/rag/llama-index) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/llama-index:r35.4.1`](https://hub.docker.com/r/dustynv/llama-index/tags) `(2024-05-23, 5.5GB)`<br>[`dustynv/llama-index:r36.2.0`](https://hub.docker.com/r/dustynv/llama-index/tags) `(2024-04-30, 6.2GB)`<br>[`dustynv/llama-index:r36.3.0`](https://hub.docker.com/r/dustynv/llama-index/tags) `(2024-05-23, 5.5GB)`<br>[`dustynv/llama-index:samples-r35.4.1`](https://hub.docker.com/r/dustynv/llama-index/tags) `(2024-06-25, 6.4GB)`<br>[`dustynv/llama-index:samples-r36.2.0`](https://hub.docker.com/r/dustynv/llama-index/tags) `(2024-06-25, 6.4GB)`<br>[`dustynv/llama-index:samples-r36.3.0`](https://hub.docker.com/r/dustynv/llama-index/tags) `(2024-06-24, 6.4GB)` |

| **`llama-index:samples`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch`](/packages/pytorch) [`llama-index:main`](/packages/rag/llama-index) [`tensorrt`](/packages/cuda/tensorrt) [`cuda-python`](/packages/cuda/cuda-python) [`pycuda`](/packages/cuda/pycuda) [`rust`](/packages/build/rust) [`jupyterlab:latest`](/packages/code/jupyterlab) [`jupyterlab:myst`](/packages/code/jupyterlab) [`ollama`](/packages/llm/ollama) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.samples`](Dockerfile.samples) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/llama-index:samples-r35.4.1`](https://hub.docker.com/r/dustynv/llama-index/tags) `(2024-06-25, 6.4GB)`<br>[`dustynv/llama-index:samples-r36.2.0`](https://hub.docker.com/r/dustynv/llama-index/tags) `(2024-06-25, 6.4GB)`<br>[`dustynv/llama-index:samples-r36.3.0`](https://hub.docker.com/r/dustynv/llama-index/tags) `(2024-06-24, 6.4GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/llama-index:r35.4.1`](https://hub.docker.com/r/dustynv/llama-index/tags) | `2024-05-23` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/llama-index:r36.2.0`](https://hub.docker.com/r/dustynv/llama-index/tags) | `2024-04-30` | `arm64` | `6.2GB` |
| &nbsp;&nbsp;[`dustynv/llama-index:r36.3.0`](https://hub.docker.com/r/dustynv/llama-index/tags) | `2024-05-23` | `arm64` | `5.5GB` |
| &nbsp;&nbsp;[`dustynv/llama-index:samples-r35.4.1`](https://hub.docker.com/r/dustynv/llama-index/tags) | `2024-06-25` | `arm64` | `6.4GB` |
| &nbsp;&nbsp;[`dustynv/llama-index:samples-r36.2.0`](https://hub.docker.com/r/dustynv/llama-index/tags) | `2024-06-25` | `arm64` | `6.4GB` |
| &nbsp;&nbsp;[`dustynv/llama-index:samples-r36.3.0`](https://hub.docker.com/r/dustynv/llama-index/tags) | `2024-06-24` | `arm64` | `6.4GB` |

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
jetson-containers run $(autotag llama-index)

# or explicitly specify one of the container images above
jetson-containers run dustynv/llama-index:samples-r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/llama-index:samples-r35.4.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag llama-index)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag llama-index) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build llama-index
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
