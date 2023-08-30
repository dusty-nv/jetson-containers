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
| &nbsp;&nbsp;&nbsp;Builds | [![`langchain_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/langchain_jp51.yml?label=langchain:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/langchain_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`huggingface_hub`](/packages/llm/huggingface_hub) [`llama_cpp`](/packages/llm/llama_cpp) |
| &nbsp;&nbsp;&nbsp;Dependants | [`langchain:samples`](/packages/llm/langchain) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`langchain:samples`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Builds | [![`langchain-samples_jp51`](https://img.shields.io/github/actions/workflow/status/dusty-nv/jetson-containers/langchain-samples_jp51.yml?label=langchain-samples:jp51)](https://github.com/dusty-nv/jetson-containers/actions/workflows/langchain-samples_jp51.yml) |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`huggingface_hub`](/packages/llm/huggingface_hub) [`llama_cpp`](/packages/llm/llama_cpp) [`langchain`](/packages/llm/langchain) [`rust`](/packages/rust) [`jupyterlab`](/packages/jupyterlab) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.samples`](Dockerfile.samples) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag langchain)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host langchain:35.2.1

```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag langchain)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag langchain) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh langchain
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
