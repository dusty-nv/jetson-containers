# jetson-copilot

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* jetson-copilot (temporary name for Ollama-LlamaIndex-based, Streamlit-enabled container)

## Starting `jetson-copilot` container

```bash
jetson-containers run $(autotag jetson-copilot)
```

This will start the `ollama` server and enter into a `bash` terminal.

## Starting "Jetson Copilot" app inside the container

First, create a directory on the host side to store Jetson related documents. The `data` directory is mounted on the container.

```
cd jetson-containers
mkdir -p ./data/documents/jetson
```


Once in the container:

```bash
streamlit run /opt/jetson-copilot/app.py
```

> Or you can start the container with additional arguments:
> ```
> jetson-containers run $(autotag jetson-copilot) bash -c '/start_ollama && streamlit run app.py'
> ```

This will start the `ollama` server and `streamlit` app for "Jetson Copilot", an AI assistant to answer any questions based on documents provided in `/data/documents/jetson` directory.

It should show something like this:

```
  You can now view your Streamlit app in your browser.

  Network URL: http://10.110.50.241:8501
  External URL: http://216.228.112.22:8501
```

### Accessing "Jetson Copilot" app 

From your browser, open the above Network URL (`http://10.110.50.241:8501`).

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`jetson-copilot`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `jetrag` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`ollama`](/packages/llm/ollama) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/jetson-copilot:r35.4.1`](https://hub.docker.com/r/dustynv/jetson-copilot/tags) `(2024-07-03, 6.3GB)`<br>[`dustynv/jetson-copilot:r36.2.0`](https://hub.docker.com/r/dustynv/jetson-copilot/tags) `(2024-07-03, 6.3GB)`<br>[`dustynv/jetson-copilot:r36.3.0`](https://hub.docker.com/r/dustynv/jetson-copilot/tags) `(2024-07-03, 6.3GB)`<br>[`dustynv/jetson-copilot:r36.4.0`](https://hub.docker.com/r/dustynv/jetson-copilot/tags) `(2024-10-13, 4.7GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/jetson-copilot:r35.4.1`](https://hub.docker.com/r/dustynv/jetson-copilot/tags) | `2024-07-03` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/jetson-copilot:r36.2.0`](https://hub.docker.com/r/dustynv/jetson-copilot/tags) | `2024-07-03` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/jetson-copilot:r36.3.0`](https://hub.docker.com/r/dustynv/jetson-copilot/tags) | `2024-07-03` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/jetson-copilot:r36.4.0`](https://hub.docker.com/r/dustynv/jetson-copilot/tags) | `2024-10-13` | `arm64` | `4.7GB` |

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
jetson-containers run $(autotag jetson-copilot)

# or explicitly specify one of the container images above
jetson-containers run dustynv/jetson-copilot:r36.4.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/jetson-copilot:r36.4.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag jetson-copilot)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag jetson-copilot) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build jetson-copilot
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
